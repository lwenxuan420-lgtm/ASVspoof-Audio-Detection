import os
import argparse
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from tqdm import tqdm
from pathlib import Path
from model import CNN2D


# =========================
# ARGPARSE (比赛标准入口)
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--csv", type=str, default=None, help="test csv file")
parser.add_argument("--audio", type=str, default=None, help="single audio file for demo")
parser.add_argument("--model", type=str, default="best_model.pth", help="model path")
parser.add_argument("--data", type=str, default="data/", help="audio root folder")

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# UTIL
# =========================
def normalize(x):
    x = os.path.basename(str(x))
    x = os.path.splitext(x)[0]
    return x.lower()


# =========================
# FEATURE EXTRACTOR
# =========================
class AudioProcessor:
    def __init__(self, sr=16000, target_len=64000):
        self.sr = sr
        self.target_len = target_len

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

    def __call__(self, wav, sr):
        wav = torch.tensor(wav).float()

        if wav.dim() == 2:
            wav = wav.mean(dim=1)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        # pad / trim
        if wav.shape[0] < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            wav = wav[:self.target_len]

        x = self.mel(wav)
        x = torch.log(x + 1e-6)
        x = (x - x.mean()) / (x.std() + 1e-6)

        return x


processor = AudioProcessor()


# =========================
# LOAD MODEL
# =========================
model = CNN2D().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()


# =========================
# GEMMA4 PLACEHOLDER (比赛加分点)
# =========================
def gemma4_explain(score):
    """
    未来可以接 Gemma4 / LLM
    现在先做 rule-based（保证可运行）
    """
    if score > 0.5:
        return "This audio shows artifacts consistent with synthetic speech generation."
    else:
        return "This audio appears to be natural human speech."


# =========================
# SINGLE AUDIO MODE (比赛强烈推荐)
# =========================
def infer_single(audio_path):
    wav, sr = sf.read(audio_path)

    x = processor(wav, sr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()

    label = "SPOOF" if score > 0.5 else "BONAFIDE"
    explanation = gemma4_explain(score)

    print("\n================ RESULT ================")
    print("File:", audio_path)
    print("Score:", round(score, 4))
    print("Prediction:", label)
    print("Explanation:", explanation)
    print("=======================================\n")


# =========================
# CSV BATCH MODE (比赛提交)
# =========================
def infer_csv(csv_path, data_root):
    df = pd.read_csv(csv_path)
    col = "audio_name" if "audio_name" in df.columns else "file_name"
    names = df[col].astype(str).tolist()

    ROOTS = [Path(data_root)]

    file_map = {}
    print("Indexing audio files...")

    for root in ROOTS:
        for path, _, files in os.walk(root):
            for f in files:
                if f.endswith((".wav", ".flac")):
                    file_map[normalize(f)] = os.path.join(path, f)

    print("Total audio files:", len(file_map))

    results = []
    missing = 0

    for name in tqdm(names):

        key = normalize(name)

        if key not in file_map:
            missing += 1
            continue

        path = file_map[key]

        try:
            wav, sr = sf.read(path)

            x = processor(wav, sr).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                score = torch.sigmoid(model(x)).item()

            results.append((name, score))

        except:
            missing += 1
            continue

    print("\nDone!")
    print("Valid:", len(results))
    print("Missing:", missing)

    out_path = "submission.csv"
    pd.DataFrame(results, columns=[col, "score"]).to_csv(out_path, index=False)

    print("Saved:", out_path)


# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":

    if args.audio:
        infer_single(args.audio)

    elif args.csv:
        infer_csv(args.csv, args.data)

    else:
        print("Please provide --audio or --csv")