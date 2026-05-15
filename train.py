import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tqdm import tqdm

from model import CNN2D


# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ================= EER =================
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.abs(fnr - fpr))]


# ================= Feature =================
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

        if wav.shape[0] < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            wav = wav[:self.target_len]

        x = self.mel(wav)
        x = torch.log(x + 1e-6)

        # ⭐统一归一化（train + inference一致）
        x = (x - x.mean()) / (x.std() + 1e-6)

        return x


# ================= Dataset =================
class AudioDataset(Dataset):
    def __init__(self, df, roots):
        self.samples = []

        for row in df.itertuples():
            for root in roots:
                path = os.path.join(root, row.audio_name)
                if os.path.exists(path):
                    self.samples.append((path, float(row.target)))
                    break

        print("Samples:", len(self.samples))
        self.processor = AudioProcessor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        wav, sr = sf.read(path)
        x = self.processor(wav, sr)

        return x.unsqueeze(0), torch.tensor(label, dtype=torch.float32)


# ================= Train =================
def train(train_roots, csv_file, epochs=15, batch_size=16):

    df = pd.read_csv(csv_file)

    # ===== split (防泄漏关键) =====
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)

    train_set = AudioDataset(train_df, train_roots)
    dev_set = AudioDataset(dev_df, train_roots)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)

    model = CNN2D().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_eer = 1.0

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}")

        # ================= EVAL =================
        model.eval()
        scores, labels = [], []

        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                out = torch.sigmoid(model(x)).cpu()

                scores.extend(out.numpy())
                labels.extend(y.numpy())

        eer = compute_eer(np.array(labels), np.array(scores))
        print(f"Epoch {epoch+1} EER: {eer:.4f}")

        # ================= SAVE BEST =================
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Saved best model")


# ================= RUN =================
if __name__ == "__main__":

    train_roots = [
        r"D:\data\train-renamepart_006\train-rename",
        r"D:\data\train-renamepart_009\train-rename"
    ]

    train_csv = r"D:\csv\train.csv"

    train(train_roots, train_csv, epochs=15, batch_size=16)