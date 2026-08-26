""" 
Antrema modelul audio doar pe fisierele de split_audio.json
Augmentarea (zgomot + time-stretch) se aplica doar pe train, niciodata pe val/test.
Modelul este evaluat pe validation la fiecare epoca
"""

import os
import json
import configparser
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

config = configparser.ConfigParser()
config.read("config.txt")

N_MFCC        = int(config["audio"]["n_mfcc"])
DURATION      = int(config["audio"]["duration"])
SAMPLE_RATE   = int(config["audio"]["sample_rate"])
EPOCHS        = int(config["audio"]["epochs"])
LEARNING_RATE = float(config["audio"]["learning_rate"])
BATCH_SIZE    = int(config["audio"]["batch_size"])
MODELS_DIR    = config["experiment"]["models_experiment_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODELS_DIR, exist_ok=True)


class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X)).float()
        self.y = torch.tensor(np.array(y)).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def extract_features_train(file_path):
    y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)
    variante = [y]

    noise = np.random.normal(0, 0.005, len(y))
    variante.append(y + noise)

    try:
        stretched = librosa.effects.time_stretch(y, rate=0.9)
        variante.append(stretched)
    except Exception:
        pass

    return [extract_mfcc(v, sr) for v in variante]


def extract_features_simplu(file_path):
    y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)
    return extract_mfcc(y, sr)


def main():
    print("=" * 60)
    print("ANTRENARE MODEL AUDIO (split experimental)")
    print("=" * 60)

    with open("split_audio.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    print("\n[TRAIN] Extragere features (cu augmentare)...")
    X_train, y_train = [], []
    for clasa, s in split.items():
        for fp in s["train"]:
            try:
                for feat in extract_features_train(fp):
                    X_train.append(feat)
                    y_train.append(clasa)
            except Exception as e:
                print(f"  [SKIP] {fp}: {e}")
    print(f"  Total exemple train (cu augmentare): {len(X_train)}")

    print("\n[VAL] Extragere features (fara augmentare)...")
    X_val, y_val = [], []
    for clasa, s in split.items():
        for fp in s["val"]:
            try:
                X_val.append(extract_features_simplu(fp))
                y_val.append(clasa)
            except Exception as e:
                print(f"  [SKIP] {fp}: {e}")
    print(f"  Total exemple validare: {len(X_val)}")

    le = LabelEncoder()
    le.fit(sorted(split.keys()))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    train_loader = DataLoader(AudioDataset(X_train, y_train_enc), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(AudioDataset(X_val, y_val_enc), batch_size=BATCH_SIZE)

    model     = AudioClassifier(26, len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    print(f"\nAntrenare: {EPOCHS} epoci, {len(le.classes_)} clase")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                preds = model(X_b).argmax(1)
                total += y_b.size(0)
                correct += (preds == y_b).sum().item()
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(total_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"  Epoch {epoch+1}/{EPOCHS}  Loss={total_loss:.4f}  Val_Acc={val_acc*100:.2f}%")

    print(f"\nCea mai buna acuratete pe validation: {best_val_acc*100:.2f}%")

    np.save(os.path.join(MODELS_DIR, "audio_classes.npy"), le.classes_)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "audio_model.pth"))
    print(f"Model salvat in {MODELS_DIR}/audio_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()