"""
Batch prediction script
"""
import argparse
import csv
import json
import librosa
import numpy as np
import os
import pathlib
import soundfile as sf
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from tqdm import tqdm

from functions import *

# ---- utils ---------------------------------------------------------------

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXT


def convert_to_wav(src: Path, target_sr=16_000) -> Path:
    """
    If the input is already WAV and the sampling rate meets the requirements, it will be returned directly;
    otherwise, soundfile + librosa resampling (or ffmpeg) will be used to save a temporary WAV.
    """
    if src.suffix.lower() == ".wav":
        try:
            info = sf.info(src)
            if info.samplerate == target_sr:
                return src
        except Exception:
            pass

    y, sr = librosa.load(src, sr=target_sr, mono=True)
    tmp_wav = Path(tempfile.gettempdir()) / f"{src.stem}_{target_sr}.wav"
    sf.write(tmp_wav, y, target_sr)
    return tmp_wav


def collect_files(path: Path):
    if path.is_file():
        return [path] if is_audio(path) else []
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and is_audio(p)]
    return []


# ---- predictor ----------------------------------------------------------
class Predictor:
    """
    mode = 'conformer' | 'whisper' | 'both'
    alpha = 权重系数，当 mode='both' 时 prob = alpha*P_conf + (1-alpha)*P_whisper
    """

    def __init__(self,
                 mode: str,
                 conformer_ckpt: Path | None,
                 whisper_ckpt: Path | None,
                 alpha: float = 0.5,
                 device: str = "cuda",
                 target_sr: int = 16000,
                 n_mfcc: int = 40,
                 hop: int = 512,
                 n_fft: int = 1024,
                 frames: int = 400,
                 threshold: float = 0.8):

        self.device = torch.device(device)
        self.mode = mode.lower()
        self.alpha = float(alpha)
        self.threshold = threshold

        # --- Conformer part ------------------------------------------------
        if self.mode in {"conformer", "both"}:
            if conformer_ckpt is None:
                raise ValueError("Conformer checkpoint must be provided.")
            self.conf_model = Conformer_plus(input_dim=n_mfcc).to(self.device)
            state = torch.load(conformer_ckpt, map_location=self.device)
            self.conf_model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
            self.conf_model.eval()
            self.mfcc_extractor = MFCCExtractor(target_sr, n_mfcc, hop, n_fft, frames)

        # --- Whisper part --------------------------------------------------
        if self.mode in {"whisper", "both"}:
            self.whi_model = WhisperEncoderForBinaryClassification().to(self.device)
            if whisper_ckpt is not None:
                state = torch.load(whisper_ckpt, map_location=self.device)
                self.whi_model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
            self.whi_model.eval()
            self.whi_extractor = WhisperExtractor()

    # ------------------ single‑file inference ----------------------------
    @torch.no_grad()
    def predict_prob(self, wav_path: Path) -> float:
        wav = convert_to_wav(wav_path)
        y, sr = librosa.load(wav, sr=None, mono=True)

        probs = []

        if self.mode in {"conformer", "both"}:
            mfcc = self.mfcc_extractor.extract_features(y, sr)  # (40,400)
            mfcc = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)
            p_conf = torch.sigmoid(self.conf_model(mfcc)).item()
            probs.append(("conf", p_conf))

        if self.mode in {"whisper", "both"}:
            feats = self.whi_extractor.extract_features(y, sr)  # (1,80,3000)
            feats = torch.from_numpy(feats).unsqueeze(0).to(self.device)
            p_whi = self.whi_model(feats).squeeze().item()
            probs.append(("whi", p_whi))

        # --- fuse ---------------------------------------------------------
        if self.mode == "conformer":
            return probs[0][1]
        if self.mode == "whisper":
            return probs[0][1]
        # both
        p_conf = dict(probs)["conf"]
        p_whi = dict(probs)["whi"]
        return self.alpha * p_conf + (1 - self.alpha) * p_whi

    def prob_to_label(self, prob: float) -> int:
        return int(prob >= self.threshold)


# ---- main ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mode", choices=["conformer", "whisper", "both"], default="conformer")

    # checkpoints
    parser.add_argument("--conformer_ckpt", type=Path, default=None)
    parser.add_argument("--whisper_ckpt", type=Path, default=None)

    # fusion weight
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Conformer when --mode both  (0–1)")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        sys.exit("❌  No valid audio files found.")

    predictor = Predictor(mode=args.mode,
                          conformer_ckpt=args.conformer_ckpt,
                          whisper_ckpt=args.whisper_ckpt,
                          alpha=args.alpha,
                          threshold=args.threshold)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file", "prob", "label"])
        for f in tqdm(files, desc="Predict"):
            try:
                prob = predictor.predict_prob(f)
                label = predictor.prob_to_label(prob)
                writer.writerow([str(f), f"{prob:.5f}", label])
            except Exception as e:
                warnings.warn(f"Skip {f}: {e}")


if __name__ == "__main__":
    print('if u want to use our whisper ckpt,make sure your torch version == 2.4.1 which we used to train the model')
    print('if u want to use this script, please run the command in following format:')
    print("python predict.py --input ./Data/11Lab/ \
                      --conformer_ckpt conformer_plus.pt \
                      --mode conformer \
                      --threshold 0.8 \
                      --csv 'results.csv'")
    main()
