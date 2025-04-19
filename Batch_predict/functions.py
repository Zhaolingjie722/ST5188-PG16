import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import WhisperModel, WhisperProcessor


class MFCCExtractor:
    def __init__(self, target_sr=16000, n_mfcc=40, hop_length=512, n_fft=1024, target_frames=400, **kwargs):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_frames = target_frames
        self.kwargs = kwargs

    def extract_features(self, audio, sr):
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            **self.kwargs
        )
        target_n_mfcc = self.n_mfcc
        target_frames = self.target_frames

        current_n_mfcc, current_frames = mfccs.shape
        if current_n_mfcc < target_n_mfcc:
            pad_amount = target_n_mfcc - current_n_mfcc
            mfccs = np.pad(mfccs, ((0, pad_amount), (0, 0)), mode='constant')
        elif current_n_mfcc > target_n_mfcc:
            mfccs = mfccs[:target_n_mfcc, :]

        if current_frames < target_frames:
            pad_amount = target_frames - current_frames
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_amount)), mode='constant')
        elif current_frames > target_frames:
            mfccs = mfccs[:, :target_frames]

        return mfccs.astype(np.float32)


# ----------------------------
# Feed Forward Module
# ----------------------------
class FeedForwardModule_plus(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.05):
        super(FeedForwardModule_plus, self).__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),  # LayerNorm
            nn.Linear(d_model, d_ff),  # Linear layer
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_ff, d_model),  # Linear layer
            nn.Dropout(dropout)  # Dropout
        )

    def forward(self, x):
        return self.ff(x)


# ----------------------------
# Multi-Head Self-Attention Module
# ----------------------------
class MultiHeadSelfAttention_plus(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.05):
        super(MultiHeadSelfAttention_plus, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        return attn_output


# ----------------------------
# Convolution Module
# ----------------------------
class ConvolutionModule_plus(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.05):
        super(ConvolutionModule_plus, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model,
                                        padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)  # (B, d_model, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        return x + residual


# ----------------------------
# Conformer Block
# ----------------------------
class ConformerBlock_plus(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.05, kernel_size=15):
        super(ConformerBlock_plus, self).__init__()
        self.ff_module1 = FeedForwardModule_plus(d_model, d_ff, dropout)
        self.self_attn = MultiHeadSelfAttention_plus(d_model, num_heads, dropout)
        self.conv_module = ConvolutionModule_plus(d_model, kernel_size, dropout)
        self.ff_module2 = FeedForwardModule_plus(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff_module1(x)
        x = x + self.self_attn(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_module2(x)
        x = self.layer_norm(x)
        return x


# ----------------------------
# Complete Conformer Network
# ----------------------------
class Conformer_plus(nn.Module):
    def __init__(self, input_dim=40, d_model=32, d_ff=64, num_heads=8, num_layers=4,
                 dropout=0.05, kernel_size=7, num_classes=1):
        """
        :param input_dim: 输入特征维度（例如 40，对应 MFCC 维度）
        :param d_model: 投影后的模型维度（例如 32）
        :param d_ff: 前馈网络隐藏层维度（例如 64）
        :param num_heads: 多头自注意力头数（例如 8）
        :param num_layers: Conformer Block 层数（例如 4 层）
        :param num_classes: 输出类别数（例如二分类时为 1）
        """
        super(Conformer_plus, self).__init__()
        # 输入投影层
        self.input_linear = nn.Linear(input_dim, d_model)

        # 堆叠多个 Conformer Block
        self.conformer_layers = nn.ModuleList([
            ConformerBlock_plus(d_model, d_ff, num_heads, dropout, kernel_size) for _ in range(num_layers)
        ])

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 输出层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 输入 x 形状为 (B, 1, 40, 400)
        if x.dim() == 4:
            x = x.squeeze(1)  # -> (B, 40, 400)

        # 转置为 (B, T, input_dim)，即 (B, 400, 40)
        x = x.transpose(1, 2)

        # 投影到 d_model 维度 -> (B, 400, d_model)
        x = self.input_linear(x)

        # 通过多个 Conformer Block
        for layer in self.conformer_layers:
            x = layer(x)

        # 转置为 (B, d_model, T) -> (B, d_model, 400)
        x = x.transpose(1, 2)

        # 全局池化
        x = self.pool(x)  # -> (B, d_model, 1)
        x = x.squeeze(-1)  # -> (B, d_model)

        # 全连接层输出
        x = self.fc(x)  # -> (B, num_classes)
        return x


class WhisperExtractor:
    def __init__(self, target_sr=16000, **kwargs):
        # the mfcc parameters are not used for whisper
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.sr = target_sr

    def extract_features(self, audio, sr):
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt", padding='max_length',
                                max_length=self.sr * 30)
        features = inputs.input_features.squeeze().numpy()  # need to check shape

        return features


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
