🚀 📄 README.md
# ASVspoof-Audio-Detection (CNN Baseline)


A lightweight and effective audio spoof / deepfake detection system based on CNN and Log-Mel spectrogram features.

Designed for:
- ASVspoof-style datasets
- Kaggle large-scale audio inference scenarios

---

## 🚀 Key Features

- End-to-end audio spoof detection pipeline
- CNN-based classifier (fast & stable)
- Log-Mel spectrogram feature extraction
- Supports large-scale datasets (100k+ samples)
- Handles missing audio entries (real-world Kaggle scenario)
- Produces submission-ready CSV output

---

## 🧠 Model Architecture


Waveform → Resampling → Log-Mel Spectrogram → CNN → Fully Connected → Score


### Network Design
- 3-layer 2D CNN
- BatchNorm + ReLU
- MaxPooling
- AdaptiveAvgPool (input-size invariant)
- Fully connected classifier

---

## 📊 Feature Extraction

Each audio file is processed as:

- Resampled to 16 kHz
- Fixed length: 64,000 samples (~4 seconds)
- 64-bin Mel spectrogram
- Log transform: log(x + 1e-6)
- Per-sample normalization (mean/std)

---

## 📁 Dataset Structure


data/
├── kaggle-audio-train-xxxx.wav
├── kaggle-audio-test-xxxx.wav

csv/
└── test.csv


### CSV Format


audio_name
kaggle-audio-test-000001.wav
kaggle-audio-test-000002.wav


---

## ⚠️ Notes

- CSV may contain audio files not present in dataset
- Missing files are automatically skipped
- This is expected in large-scale Kaggle-style datasets

---

## ⚙️ Installation

```bash
pip install torch torchaudio numpy pandas soundfile scikit-learn tqdm
🏋 Training
python main.py

Training settings:

Train/Dev split: 90/10
Loss: BCEWithLogitsLoss
Optimizer: Adam (lr=1e-3)
Metric: EER
Best model saved as best_model.pth
🚀 Inference
python inference.py

Output:

submission.csv

Format:

audio_name,score
kaggle-audio-test-000001.wav,0.7321
kaggle-audio-test-000002.wav,0.1284
📂 Project Structure
train.py            # training pipeline
model.py           # CNN model definition
inference.py       # inference script
best_model.pth     # trained checkpoint
README.md
requirements.txt
.gitignore
📊 Results
EER: ~0.01–0.05 (dataset dependent)
Stable convergence within 10–15 epochs
Strong generalization on unseen samples
🔬 Applications
ASVspoof baseline research
Audio deepfake detection
Kaggle audio classification
Lightweight edge inference systems
📌 Reproducibility
python main.py
python inference.py
📄 License

Academic use only.