import cv2
import gc
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tempfile
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import (roc_curve, recall_score, precision_score, accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report, precision_recall_curve)
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
from transformers import WhisperModel, WhisperProcessor, WhisperConfig, Trainer, TrainingArguments, PreTrainedModel, \
    AutoConfig

MODEL = 'xxx'  # where to save, and model prefix
META = 'xxx.csv'  # the data meta file

EVAL_ONLY = False
VALIDATE = False  # 一般关闭
EPOCH = 9

LR = 0.001

THRESH = 0.5

LJ_REPEAT = -1

TRAIN = ['itw', 'gen', 'vox',
         'diffusion-based']  # ['itw', 'gen',  'vox', 'diffusion-based'] 去掉diffusion-based train的话应该就能得到不太好的eer泛化
OUT_TEST = ['11lab']  # ['itw', 'gen', 'vox', 'diffusion-based', '11lab']
IN_TEST = []  # ['itw', 'gen', 'vox', 'diffusion-based']

print("train: ", TRAIN)
print("out_test: ", OUT_TEST)
print("in_test: ", IN_TEST, flush=True)

print(MODEL, META, flush=True)
print("**************************\n")


class CSVAudioLoader:
    def __init__(self, max_workers=96):
        """
        初始化 CSVAudioLoader，仅用于读取 CSV 文件中的音频路径和元数据。
        :param max_workers: 多线程时最大线程数，默认 96
        """
        self.max_workers = max_workers

    def load_audio_info_from_csv(self, csv_file, allowed_set=None, allowed_data_source="all"):
        """
        从 CSV 文件中加载音频数据的相关信息，不读取音频数据本身。
        CSV 文件中应包含以下字段：
          - absolute_path: 音频文件的绝对路径
          - data_source: 数据源（例如 "itw"）
          - speaker: 说话人信息
          - Set: 数据集划分（例如 "train", "test", "in-test"）
          - label: 分类标签
        :param csv_file: CSV 文件路径
        :param allowed_set: 允许加载的 Set 值列表，若为 None，则不进行过滤
        :param allowed_data_source: 如果设置为 "all" 则不进行 data_source 过滤，否则必须为列表，例如 ["itw"]
        :return: 返回一个列表，每个元素为 (absolute_path, speaker, label)
        """
        df = pd.read_csv(csv_file)
        if allowed_set is not None:
            df = df[df["Set"].isin(allowed_set)]
        if allowed_data_source != "all":
            df = df[df["data_source"].isin(allowed_data_source)]

        if LJ_REPEAT != -1:
            repeat = df[df['data_source'] == 'LJSpeech']
            repeat = pd.concat([repeat] * (LJ_REPEAT - 1), ignore_index=True)
            df = pd.concat([df, repeat], ignore_index=True)

        print("*** Train includes ***")
        print(df['data_source'].value_counts(), flush=True)

        # exit()

        def load_info(row):
            file_path = row["relative_path"]
            speaker = row["speaker"]
            label = int(row["label"])  # 这里转换为 int 类型
            src = row["data_source"]
            return (file_path, speaker, label, src)

        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(load_info, row): idx for idx, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading audio info from CSV"):
                result = future.result()
                if result is not None:
                    records.append(result)
        return records


# ----读取audio而不是path
class CSVAudioLoader1:
    def __init__(self, target_sr=16000, max_workers=96):
        self.target_sr = target_sr
        self.max_workers = max_workers

    def load_audio_from_csv(self, csv_file, allowed_set=None, allowed_data_source="all"):
        df = pd.read_csv(csv_file)
        # 根据 allowed_set 过滤数据（假设 CSV 中的列名为 "Set"）
        if allowed_set is not None:
            df = df[df["Set"].isin(allowed_set)]
        # 根据 allowed_data_source 过滤数据（当其不等于 "all" 时）
        if allowed_data_source != "all":
            df = df[df["data_source"].isin(allowed_data_source)]

        print("*** Test includes ***")
        print(df['data_source'].value_counts(), flush=True)

        def load_audio(row):
            file_path = row["relative_path"]
            try:
                audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
                return (file_path, audio, sr, row["speaker"], row['label'], row['data_source'])
            except Exception as e:
                print(f"Error loading audio from {file_path}: {e}")
                return None

        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(load_audio, row): idx for idx, row in df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading audio from CSV"):
                result = future.result()
                if result is not None:
                    records.append(result)
        return records


processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


class MFCCExtractor:
    def __init__(self, target_sr=16000, n_mfcc=60, hop_length=512, n_fft=1024, target_frames=400, max_workers=32,
                 **kwargs):
        """
        初始化 MFCC 提取器，可自定义 MFCC 参数，并将输出统一调整为 (n_mfcc, target_frames)
        :param target_sr: sr 采样频率，默认16000
        :param n_mfcc: MFCC 数量，默认 60
        :param hop_length: 帧移，默认 512
        :param n_fft: FFT 点数，默认 1024
        :param target_frames: 目标时间帧数，默认 400（仅用于 extract_features 中的 resize）
        :param max_workers: 多线程时的线程数，默认 80
        :param kwargs: 其他参数传给 librosa.feature.mfcc
        """
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_frames = target_frames
        self.max_workers = max_workers
        self.kwargs = kwargs

    def extract_features(self, audio, sr):
        """
        对单个音频信号提取 MFCC 特征，并通过 cv2.resize 调整输出为固定尺寸 (n_mfcc, target_frames)
        :param audio: 音频信号 (numpy 数组)
        :param sr: 采样率
        :return: MFCC 特征矩阵 (numpy 数组)，shape 为 (n_mfcc, target_frames)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            **self.kwargs
        )
        # resized_mfcc = cv2.resize(mfccs.astype(np.float32), (self.target_frames, self.n_mfcc), interpolation=cv2.INTER_LINEAR)
        # return resized_mfcc

        # 目标尺寸
        target_n_mfcc = self.n_mfcc
        target_frames = self.target_frames

        # 处理频率维度 (n_mfcc)
        current_n_mfcc, current_frames = mfccs.shape
        if current_n_mfcc < target_n_mfcc:
            # 不足则在下方补 0
            pad_amount = target_n_mfcc - current_n_mfcc
            mfccs = np.pad(mfccs, ((0, pad_amount), (0, 0)), mode='constant')
        elif current_n_mfcc > target_n_mfcc:
            # 多余则裁剪前面部分
            mfccs = mfccs[:target_n_mfcc, :]

        # 处理时间帧数 (target_frames)
        if current_frames < target_frames:
            pad_amount = target_frames - current_frames
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_amount)), mode='constant')
        elif current_frames > target_frames:
            mfccs = mfccs[:, :target_frames]

        return mfccs.astype(np.float32)

    def extract_batch(self, audio_data):
        """
        对一批音频数据进行批量 MFCC 特征提取，使用多线程方式。
        每个音频数据应为至少包含以下字段的元组：
          (file_path, audio, sr, data_source, speaker, Set, label)
        提取后返回 (mfcc_features, speaker, label) 供后续防止 speaker 泄露和标签保存使用。
        :param audio_data: 列表，每个元素至少包含 (file_path, audio, sr, speaker, label)
        :return: 列表，每个元素为 (mfcc_features, speaker, label)
        """

        def process_row(row):
            # row 中的顺序: 0: file_path, 1: audio, 2: sr, 3: speaker 4: label
            audio = row[1]
            sr = row[2]
            # mfcc_features = self.extract_features(audio, sr)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding='max_length', max_length=sr * 30)
            mfcc_features = inputs.input_features.squeeze().numpy()
            speaker = row[3]
            label = row[4]
            src = row[5]
            return (mfcc_features, speaker, label, src)

        features = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_row, row): row for row in audio_data}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting MFCC features"):
                try:
                    result = future.result()
                    features.append(result)
                except Exception as e:
                    print(f"Error extracting features: {e}")
        return features


class AudioMFCCDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        if index < 0 or index >= len(self.samples):
            raise IndexError("Index {} out of range".format(index))

        mfcc_features, speaker, label, src = self.samples[index]

        if self.transform:
            mfcc_features = self.transform(mfcc_features)

        features = torch.tensor(mfcc_features, dtype=torch.float32)

        # 确保 features 为 3D: (channels, n_mfcc, target_frames)
        # if features.dim() == 2:
        #     features = features.unsqueeze(0)
        # elif features.dim() == 3 and features.shape[0] != 1:
        #     features = features[0:1, ...]

        label = torch.tensor(label, dtype=torch.long)
        return features, speaker, label, src

    def get_group_splits(self, n_splits=5, random_state=None):
        """
        GroupKFold 
        :param n_splits: 拆分的 Fold 数量，默认 5
        :param random_state: 随机种子，GroupKFold 本身不支持随机种子，如果需要随机打乱，请先自行打乱数据。
        :return: 返回一个列表，每个元素为 (train_idx, valid_idx)
        """
        n_samples = len(self)
        indices = np.arange(n_samples)
        # 每个样本的 speaker 信息存放在 self.samples 中，格式为 (mfcc_features, speaker, label)
        groups = [self.samples[i][1] for i in range(n_samples)]
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(indices, groups=groups))
        return splits


class LazyMFCCDataset(Dataset):

    def __init__(self, records, target_sr=16000, n_mfcc=40, hop_length=512, n_fft=1024, target_frames=400,
                 shuffle=True, cache_enabled=True, num_workers=32, batch_size=32):
        """
        :param records: 包含文件路径的列表，每个元素是 (absolute_path, speaker, label)
        :param target_sr: 目标采样率
        :param n_mfcc: 提取的 MFCC 维度
        :param hop_length: MFCC 提取的帧移
        :param n_fft: FFT 点数
        :param target_frames: 目标时间帧数
        :param shuffle: 是否打乱数据
        :param cache_enabled: 是否缓存已计算的 MFCC
        :param num_workers: 多线程加载时的线程数（用于批量加载）
        :param batch_size: 批量加载时的样本数
        """
        self.records = records
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_frames = target_frames
        self.shuffle = shuffle
        self.cache_enabled = cache_enabled
        self.num_workers = num_workers
        self.batch_size = batch_size

        # **缓存 MFCC 特征，避免重复计算**
        self.cache = {}
        # 添加线程锁，确保缓存操作线程安全
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.records)

    def _extract_features(self, audio):
        """
        **从音频信号中提取 MFCC 并调整尺寸**
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        # 调整尺寸到 (n_mfcc, target_frames)
        resized_mfcc = cv2.resize(mfccs.astype(np.float32), (self.target_frames, self.n_mfcc),
                                  interpolation=cv2.INTER_LINEAR)
        return resized_mfcc

    def _extract_features1(self, audio):
        """
        **从音频信号中提取 MFCC 并通过 padding 与裁剪调整尺寸到 (n_mfcc, target_frames)**
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        # 目标尺寸
        target_n_mfcc = self.n_mfcc
        target_frames = self.target_frames

        # 处理频率维度 (n_mfcc)
        current_n_mfcc, current_frames = mfccs.shape
        if current_n_mfcc < target_n_mfcc:
            # 不足则在下方补 0
            pad_amount = target_n_mfcc - current_n_mfcc
            mfccs = np.pad(mfccs, ((0, pad_amount), (0, 0)), mode='constant')
        elif current_n_mfcc > target_n_mfcc:
            # 多余则裁剪前面部分
            mfccs = mfccs[:target_n_mfcc, :]

        # 处理时间帧数 (target_frames)
        if current_frames < target_frames:
            pad_amount = target_frames - current_frames
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_amount)), mode='constant')
        elif current_frames > target_frames:
            mfccs = mfccs[:, :target_frames]

        return mfccs.astype(np.float32)

    def _load_audio_and_compute_mfcc(self, record):
        absolute_path, speaker, label, src = record

        # **线程安全的缓存检查**
        with self.lock:
            if self.cache_enabled and absolute_path in self.cache:
                return self.cache[absolute_path]

        try:
            # **加载音频**
            audio, sr = librosa.load(absolute_path, sr=self.target_sr, mono=True)

            # **提取 MFCC**
            # mfcc_features = self._extract_features1(audio)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding='max_length', max_length=sr * 30)
            mfcc_features = inputs.input_features.squeeze().numpy()
            # print(mfcc_features.shape)

            # **线程安全地存入缓存**
            with self.lock:
                if self.cache_enabled:
                    self.cache[absolute_path] = (mfcc_features, speaker, label, src)

            return mfcc_features, speaker, label, src

        except Exception as e:
            print(f"Error loading {absolute_path}: {e}")
            return None

    def __getitem__(self, index):
        """
        获取指定索引处的单个样本数据
        返回顺序为 (features, speaker, label)
        """
        record = self.records[index]
        result = self._load_audio_and_compute_mfcc(record)
        # print(result)

        # **如果加载失败，随机抽取其他样本**
        # while result is None:
        #     index = np.random.randint(0, len(self.records))
        #     result = self._load_audio_and_compute_mfcc(self.records[index])

        # 单个样本返回的特征为 (n_mfcc, target_frames)，这里转换为 (1, n_mfcc, target_frames)
        features, speaker, label, src = result
        features = torch.tensor(features, dtype=torch.float32)
        # if features.dim() == 2:
        #     features = features.unsqueeze(0)
        # elif features.dim() == 3 and features.shape[0] != 1:
        #     features = features[0:1, ...]
        # 注意：label 保持原始数据类型，后续 collate 函数会转换成 tensor
        return features, speaker, label, src

    def get_group_splits(self, n_splits=5):
        """
        **使用 GroupKFold 根据 speaker 进行分组**
        """
        n_samples = len(self)
        indices = np.arange(n_samples)
        groups = [self.records[i][1] for i in range(n_samples)]  # 使用 speaker 进行分组
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(indices, groups=groups))
        return splits


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # CNN 部分：输入 (batch, 1, 50, 400)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.2)

        # 经过两层卷积+池化：
        # conv1: (32, 60, 400) -> pool -> (32, 30, 200)
        # conv2: (64, 30, 200) -> pool -> (64, 15, 100)
        # 使用自适应平均池化，将输出尺寸降至 (batch, 64, 1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout3 = nn.Dropout(0.5)
        # 全连接层，将 64 维特征映射到 1 个输出
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, 1, 60, 400)
        x = torch.relu(self.conv1(x))  # -> (batch, 32, 60, 400)
        x = self.pool(x)  # -> (batch, 32, 30, 200)
        x = self.dropout1(x)

        x = torch.relu(self.conv2(x))  # -> (batch, 64, 30, 200)
        x = self.pool(x)  # -> (batch, 64, 15, 100)
        x = self.dropout2(x)

        # 自适应平均池化将特征图降维到 (batch, 64, 1, 1)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # -> (batch, 64)
        x = self.dropout3(x)
        x = self.fc(x)  # -> (batch, 1)
        x = torch.sigmoid(x)
        return x


class WhisperEncoderForBinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # this line for pretained whisper
        self.encoder = WhisperModel.from_pretrained("openai/whisper-tiny.en").encoder

        # this two lines for training from scratch
        # configuration = WhisperConfig()
        # self.encoder = WhisperModel(configuration).encoder

        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)  # 二分类

    def forward(self, input_features):
        encoder_outputs = self.encoder(input_features).last_hidden_state
        pooled_output = encoder_outputs.mean(dim=1)  # 简单的池化操作，取平均值
        logits = self.fc(pooled_output)
        x = torch.sigmoid(logits)
        # print(x.shape)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--device_available: {device}--", flush=True)


class CNNTrainer:
    """
    Trainer 类整合了模型定义、训练过程、早停、checkpoint 保存以及 wandb 可视化。
    使用 CNNClassifier 模型进行训练。
    """

    def __init__(self, train_dataloader=None, val_dataloader=None, device=None, lr=0.001, patience=5,
                 checkpoint_path="Workbench_v1.pt", project_name="DeepFake4Celeb_workbench_V1"):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.patience = patience

        # 使用 CNNClassifier 模型
        # self.model = CNNClassifier().to(self.device)
        self.model = WhisperEncoderForBinaryClassification().to(self.device)

        # 定义损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # # 初始化 wandb
        # wandb.init(project=project_name)
        # wandb.watch(self.model, log="all")

        # 用于保存最佳模型指标（以验证 loss 为标准）
        self.best_val_loss = float('inf')
        self.no_improve_count = 0

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=False)
        # 假设 DataLoader 返回 (features, labels, speaker)，这里忽略 speaker
        for features, speakers, labels, src in progress_bar:
            features = features.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (outputs >= 0.5).long()
            running_correct += (preds == labels.long()).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        return epoch_loss, epoch_acc

    def validate_one_epoch(self):
        if self.val_dataloader is None or not VALIDATE:
            return None, None

        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.val_dataloader, desc="Validating", leave=False)
        with torch.no_grad():
            for features, speakers, labels, src in progress_bar:
                features = features.to(self.device)
                labels = labels.to(self.device).float()

                outputs = self.model(features).squeeze(1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                preds = (outputs >= 0.5).long()
                running_correct += (preds == labels.long()).sum().item()
                total_samples += labels.size(0)

                progress_bar.set_postfix(loss=loss.item())

        val_loss = running_loss / total_samples
        val_acc = running_correct / total_samples
        return val_loss, val_acc

    def save_model(self, filepath):
        """保存当前模型参数到指定文件"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """加载模型参数"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")

    def predict(self, test_dataloader):
        """
        对测试数据进行预测，返回预测结果列表
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _, _ in test_dataloader:
                features = features.to(self.device)
                outputs = self.model(features).squeeze(1)
                preds = (outputs >= 0.5).long().cpu().numpy()
                predictions.extend(preds)
        return predictions

    def fit(self, num_epochs=20):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate_one_epoch()

            if val_loss is not None:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # 若验证 loss 改进，则保存最佳模型，并重置未改进计数器；否则累计未改进轮数
            checkpoint = self.checkpoint_path + '_' + str(epoch) + '.pt'
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improve_count = 0
                    self.save_model(checkpoint)
                else:
                    self.no_improve_count += 1
                    print(f"No improvement count: {self.no_improve_count}/{self.patience}")
                    if self.no_improve_count >= self.patience:
                        print("Early stopping triggered.")
                        break
            if not VALIDATE:
                self.save_model(checkpoint)

            log_dict = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "epoch": epoch + 1
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
                log_dict["val_acc"] = val_acc

            # wandb.log(log_dict)


class AudioDataModule:
    """
    封装数据加载流程，包括：
      1. 从 CSV 加载音频路径及元数据；
      2. 构建 LazyMFCCDataset；
      3. 根据 speaker 进行 GroupKFold 划分子集；
      4. 构建 train 和 valid DataLoader；
    """

    def __init__(self, csv_file, allowed_set=["train"], allowed_data_source="all",
                 target_sr=16000, n_mfcc=40, hop_length=512, n_fft=1024, target_frames=400,
                 batch_size=128, num_workers_csv=96, num_workers_dataset=0):
        """
        :param csv_file: CSV 文件路径
        :param allowed_set: 允许加载的 Set 值列表
        :param allowed_data_source: 数据源过滤条件
        :param target_sr: 目标采样率
        :param n_mfcc: MFCC 维度
        :param hop_length: 帧移
        :param n_fft: FFT 点数
        :param target_frames: 目标时间帧数
        :param batch_size: DataLoader 的 batch_size
        :param num_workers_csv: CSVAudioLoader 的线程数
        :param num_workers_dataset: DataLoader 的 num_workers
        """
        self.csv_file = csv_file
        self.allowed_set = allowed_set
        self.allowed_data_source = allowed_data_source
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_frames = target_frames
        self.batch_size = batch_size
        self.num_workers_csv = num_workers_csv
        self.num_workers_dataset = num_workers_dataset

        self.train_loader = None
        self.valid_loader = None
        self.dataset = None

    def setup(self):

        def custom_collate_fn(batch):
            """
            自定义 collate_fn 以支持 batch 读取,返回顺序为 (features, speakers, labels)
            假设每个样本的 features 形状为 (1, n_mfcc, target_frames)
            """
            features, speakers, labels, src = zip(*batch)
            features = torch.stack(features, dim=0)
            try:
                labels = torch.tensor([int(l) for l in labels], dtype=torch.long)
            except Exception as e:
                print(f"Error in labels conversion: {labels} -> {e}")
                raise ValueError(f"Label conversion error: {labels}")
            return features, speakers, labels, src

        # 1. 从 CSV 加载音频路径和元数据
        audio_loader = CSVAudioLoader(max_workers=self.num_workers_csv)
        records = audio_loader.load_audio_info_from_csv(
            csv_file=self.csv_file,
            allowed_set=self.allowed_set,
            allowed_data_source=self.allowed_data_source
        )
        # 2. 构建 LazyMFCCDataset
        self.dataset = LazyMFCCDataset(
            records=records,
            target_sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            target_frames=self.target_frames
        )
        # 3. 使用 GroupKFold 划分数据集
        splits = self.dataset.get_group_splits(n_splits=5)
        train_idx, valid_idx = splits[0]
        train_subset = Subset(self.dataset, train_idx)
        valid_subset = Subset(self.dataset, valid_idx)
        # 4. 创建 DataLoader
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers_dataset,
            collate_fn=custom_collate_fn
        )
        self.valid_loader = DataLoader(
            valid_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_dataset,
            collate_fn=custom_collate_fn
        )

    def get_loaders(self):
        if self.train_loader is None or self.valid_loader is None:
            self.setup()
        return self.train_loader, self.valid_loader


data_module = AudioDataModule(
    csv_file=META,  # --meta_file
    allowed_set=["train"],  # 训练还是测试
    allowed_data_source=TRAIN,
    # ['itw', 'gen', 'LJSpeech', 'vox', 'diffusion-based'] 去掉diffusion-based train的话应该就能得到不太好的eer泛化
    target_sr=16000,
    n_mfcc=40,  # --mfcc采样器个数,越大提取的特征维度越多,内存相应增大 (这个可以改改40-80)
    hop_length=512,
    n_fft=1024,  # 1024/2048 采样窗口大小
    target_frames=400,  # 目标维度,这里固定为400帧 10s的数据按上面的参数大概就是360多帧
    batch_size=128,  # 训练batch_size
    num_workers_csv=96,  # 多线程读取csv的
    num_workers_dataset=0
)

data_module.setup()

train_loader, valid_loader = data_module.get_loaders()

for batch in train_loader:
    features, speakers, labels, src = batch
    print(f"Feature Shape: {features.shape}, Labels Shape: {len(speakers)}, Speakers: {len(speakers)}, Src: {len(src)}",
          flush=True)
    break  # 打印第一个 batch检查input shape

trainer = CNNTrainer(
    train_dataloader=train_loader,
    val_dataloader=valid_loader,
    device=device,
    lr=LR,
    patience=5,
    checkpoint_path=MODEL,
    project_name="deepfake-detection"
)

if not EVAL_ONLY:
    trainer.fit(num_epochs=EPOCH)


def evaluate_model(trainer, test_dataloader):
    """
    params：
    trainer : CNNTrainer  实例,包含训练好模型和设备信息
    test_dataloader : DataLoader  测试数据的 DataLoader,建议返回格式为 (features, speaker, labels)
    ------    
    return：
    metrics : dict  包含各项评估指标的字典，方便 wandb.log() 记录
    """

    trainer.model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    misclassified_speakers = {}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
            # 如果返回的是 (features, speaker, labels) 则解包为 features, speaker, labels
            if len(batch) == 4:
                features, speakers, labels, src = batch
            else:
                features, labels = batch

            # 如果 labels 不是 Tensor，则尝试转换为 Tensor
            if not isinstance(labels, torch.Tensor):
                try:
                    # 如果 labels 是列表或 tuple，且元素为字符串，则先转换为 float
                    if isinstance(labels, (list, tuple)) and isinstance(labels[0], str):
                        labels = torch.tensor([float(l) for l in labels], dtype=torch.float32)
                    else:
                        labels = torch.tensor(labels, dtype=torch.float32)
                except Exception as e:
                    print("Error converting labels to tensor:", e)
                    continue

            features = features.to(trainer.device)
            labels = labels.to(trainer.device).float()
            labels_np = labels.cpu().numpy()

            outputs = trainer.model(features).squeeze(1)
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.4).long().cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            for idx, sp in enumerate(speakers):
                if sp is not None and preds[idx] != labels_np[idx]:
                    misclassified_speakers[sp] = misclassified_speakers.get(sp, 0) + 1

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    recall_val = recall_score(all_labels, all_preds)
    precision_val = precision_score(all_labels, all_preds)
    accuracy_val = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    average_precision = average_precision_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    # 计算 Precision-Recall 曲线数据
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)

    # 计算 ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.nanargmin(abs_diffs)
    eer = fpr[idx_eer]

    print("Accuracy: {:.4f}".format(accuracy_val))
    print("Precision: {:.4f}".format(precision_val))
    print("Recall: {:.4f}".format(recall_val))
    print("AUC: {:.4f}".format(auc_score))
    print("Average Precision (PR AUC): {:.4f}".format(average_precision))
    print("EER: {:.4f}".format(eer))
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)
    print("Misclassified Speakers:", misclassified_speakers)

    # 绘制 Precision-Recall 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall_curve, precision_curve, marker='.', label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    # 绘制 ROC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label="ROC Curve (AUC = {:.4f})".format(auc_score))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    metrics = {
        "accuracy": accuracy_val,
        "precision": precision_val,
        "recall": recall_val,
        "auc": auc_score,
        "average_precision": average_precision,
        "eer": eer,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "misclassified_speakers": misclassified_speakers
    }
    return metrics


import pandas as pd

meta = pd.read_csv(META)
print(list(meta[meta['data_source'] == 'diffusion-based'].speaker.unique()))
print(meta.columns, flush=True)


class EvaluateModelBySpeaker:
    def __init__(self, trainer, test_dataloader, threshold=0.4, target_speakers=None, target_source=None):
        """
        :param trainer: 已训练好的模型封装类（例如 CNNTrainer），包含 model 和 device 属性
        :param test_dataloader: 测试数据的 DataLoader，返回 (features, speakers, labels)
        :param threshold: 二分类决策阈值，默认为 0.4
        :param target_speakers: 需要单独评估的 speaker 类型集合
        """
        assert target_speakers is None or target_source is None
        self.trainer = trainer
        self.test_dataloader = test_dataloader
        self.threshold = threshold
        if target_source is None:
            self.target_source = {}
        else:
            self.target_source = target_source
        if target_speakers is None:
            self.target_speakers = {}
        else:
            self.target_speakers = target_speakers

    def map_speaker(self, sp):
        """
        如果 sp 在 target_speakers 中，返回 sp，否则返回 "others"
        """
        return sp if sp in self.target_speakers else "others"

    def map_src(self, sp):
        """
        如果 sp 在 target_speakers 中，返回 sp，否则返回 "others"
        """
        return sp if sp in self.target_source else "others"

    def evaluate(self):
        """
        按 speaker 分组，分别评估模型
        :return: dict, 键为组别，值为对应的评估指标字典
        """
        groups_data = {}
        for sp in self.target_speakers:
            groups_data[sp] = {"preds": [], "probs": [], "labels": []}
        for sp in self.target_source:
            groups_data[sp] = {"preds": [], "probs": [], "labels": []}
        groups_data["others"] = {"preds": [], "probs": [], "labels": []}
        groups_data["all"] = {"preds": [], "probs": [], "labels": []}

        self.trainer.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
                if len(batch) == 4:
                    features, speakers, labels, src = batch
                else:
                    features, labels = batch
                    speakers = [None] * features.shape[0]

                if not isinstance(labels, torch.Tensor):
                    try:
                        if isinstance(labels, (list, tuple)) and isinstance(labels[0], str):
                            labels = torch.tensor([float(l) for l in labels], dtype=torch.float32)
                        else:
                            labels = torch.tensor(labels, dtype=torch.float32)
                    except Exception as e:
                        print("Error converting labels to tensor:", e)
                        continue

                features = features.to(self.trainer.device)
                labels = labels.to(self.trainer.device).float()
                labels_np = labels.cpu().numpy()

                outputs = self.trainer.model(features).squeeze(1)
                probs = outputs.cpu().numpy()
                preds = (outputs >= self.threshold).long().cpu().numpy()

                if self.target_speakers:
                    for idx, sp in enumerate(speakers):
                        group = self.map_speaker(sp) if sp is not None else "others"
                        groups_data[group]["preds"].append(preds[idx])
                        groups_data[group]["probs"].append(probs[idx])
                        groups_data[group]["labels"].append(labels_np[idx])
                elif self.target_source:
                    for idx, sp in enumerate(src):
                        group = self.map_src(sp) if sp is not None else "others"
                        groups_data[group]["preds"].append(preds[idx])
                        groups_data[group]["probs"].append(probs[idx])
                        groups_data[group]["labels"].append(labels_np[idx])

                        groups_data["all"]["preds"].append(preds[idx])
                        groups_data["all"]["probs"].append(probs[idx])
                        groups_data["all"]["labels"].append(labels_np[idx])

        results = {}
        for group, data in groups_data.items():
            preds = np.array(data["preds"])
            probs = np.array(data["probs"])
            labels = np.array(data["labels"])
            if len(labels) == 0:
                print(f"Group {group} 没有数据。")
                continue

            # 如果当前组只有一种类别，则部分指标无法计算
            unique_classes = np.unique(labels)
            if len(unique_classes) < 2:
                print(f"Group {group} 仅包含类别: {unique_classes}. 部分指标将无法计算。")
                recall_val = recall_score(labels, preds, zero_division=0)
                precision_val = precision_score(labels, preds, zero_division=0)
                accuracy_val = accuracy_score(labels, preds)
                auc_score = np.nan
                average_precision = average_precision_score(labels, probs)
            else:
                recall_val = recall_score(labels, preds, zero_division=0)
                precision_val = precision_score(labels, preds, zero_division=0)
                accuracy_val = accuracy_score(labels, preds)
                auc_score = roc_auc_score(labels, probs)
                average_precision = average_precision_score(labels, probs)

            conf_matrix = confusion_matrix(labels, preds)
            report = classification_report(labels, preds, zero_division=0)

            # ROC、PR曲线及EER计算，仅在存在两类时计算
            if len(unique_classes) < 2:
                eer = np.nan
                precision_curve, recall_curve = None, None
                fpr, tpr = None, None
            else:
                precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
                fpr, tpr, thresholds = roc_curve(labels, probs)
                fnr = 1 - tpr
                abs_diffs = np.abs(fpr - fnr)
                # 如果 abs_diffs 全为 NaN，则 EER 置为 NaN
                if np.isnan(abs_diffs).all():
                    eer = np.nan
                else:
                    idx_eer = np.nanargmin(abs_diffs)
                    eer = fpr[idx_eer]

            # 绘制曲线
            if precision_curve is not None and recall_curve is not None:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(recall_curve, precision_curve, marker='.', label="PR Curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"{group} - Precision-Recall Curve")
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 2, 2)
                plt.plot(fpr, tpr, marker='.', label=f"ROC Curve (AUC = {auc_score:.4f})")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"{group} - ROC Curve")
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.show()

            results[group] = {
                "accuracy": accuracy_val,
                "precision": precision_val,
                "recall": recall_val,
                "auc": auc_score,
                "average_precision": average_precision,
                "eer": eer,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": report
            }

            print(f"Group: {group}")
            print("Accuracy: {:.4f}".format(accuracy_val))
            print("Precision: {:.4f}".format(precision_val))
            print("Recall: {:.4f}".format(recall_val))
            print("AUC: {:.4f}".format(auc_score))
            print("Average Precision (PR AUC): {:.4f}".format(average_precision))
            print("EER: {:.4f}".format(eer))
            print("Confusion Matrix:")
            print(conf_matrix)
            print("Classification Report:")
            print(report)
            print("-" * 50)

        return results


audio_loader1 = CSVAudioLoader1()  # 预测的话就一个批次全部读取进来,不用逐个转换了
mfcc_extractor = MFCCExtractor(n_mfcc=40)

out_test_audio = audio_loader1.load_audio_from_csv(csv_file=META, allowed_set=["test"], allowed_data_source=OUT_TEST)
in_test_audio = audio_loader1.load_audio_from_csv(csv_file=META, allowed_set=["in-test"], allowed_data_source=IN_TEST)

mfcc4test_out = mfcc_extractor.extract_batch(out_test_audio)
print('*** out extracted', flush=True)
mfcc4test_in = mfcc_extractor.extract_batch(in_test_audio)
print('*** in extracted', flush=True)

mfcc4test_dataset_in = AudioMFCCDataset(mfcc4test_in)
mfcc4test_dataset_out = AudioMFCCDataset(mfcc4test_out)

test_dataloader_in = DataLoader(mfcc4test_dataset_in, batch_size=32, shuffle=False)
test_dataloader_out = DataLoader(mfcc4test_dataset_out, batch_size=32, shuffle=False)

checkpoint = [f"{MODEL}_{i}.pt" for i in range(EPOCH)]

for model in checkpoint:
    print('\n\n**** start evaluate', model, flush=True)
    trainer.load_model(model)

    # metrics = evaluate_model(trainer, test_dataloader_in)
    # print(metrics)
    # # wandb.log(metrics)

    # metrics = evaluate_model(trainer, test_dataloader_out)
    # print(metrics)
    # wandb.log(metrics)

    print('*********start test on ood************')
    evaluator = EvaluateModelBySpeaker(trainer, test_dataloader_out, threshold=THRESH, target_source=OUT_TEST)
    results = evaluator.evaluate()
    print('out', results)

    print('**********start test on in-domain***********')
    evaluator = EvaluateModelBySpeaker(trainer, test_dataloader_in, threshold=THRESH, target_source=IN_TEST)
    results = evaluator.evaluate()
    print('in', results)
