""" 
Experimentul 1 din plan - Clasificare Audio-only
Evaluam modelul audio pe subsetul TEST din split_audio.json
(acestea sunt fisiere pe care modelul nostru nu le-a vazut nici la train, nici la validare)
"""



import os
import json
import configparser
import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

config = configparser.ConfigParser()
config.read("config.txt")

N_MFCC      = int(config["audio"]["n_mfcc"])
DURATION    = int(config["audio"]["duration"])
SAMPLE_RATE = int(config["audio"]["sample_rate"])
MODELS_DIR  = config["experiment"]["models_experiment_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


def main():
    print("=" * 60)
    print("EXPERIMENT 1 — Audio-only (pe setul de test)")
    print("=" * 60)

    classes = np.load(os.path.join(MODELS_DIR, "audio_classes.npy"), allow_pickle=True)
    model = AudioClassifier(26, len(classes))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "audio_model.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()

    with open("split_audio.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    y_true, y_pred = [], []
    detalii = [] 

    for clasa, s in split.items():
        test_files = s["test"]

        for fp in test_files:
            try:
                if not os.path.exists(fp):
                    print(f"  [NOT FOUND] Calea nu există: {fp}")
                    continue

                features = extract_mfcc(fp)
                tensor = torch.tensor(features).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(tensor)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred = classes[np.argmax(probs)]

                y_true.append(clasa)
                y_pred.append(pred)
                detalii.append({
                    "fisier": fp,
                    "clasa_reala": clasa,
                    "clasa_prezisa": pred,
                    "probabilitati": probs.tolist(),
                })
            except Exception as e:
                print(f"  [SKIP] {fp}: {e}")

    if len(y_true) == 0:
        print("\n[EROARE] Nu s-a putut evalua niciun fișier. Verifică structura fișierului JSON!")
        return

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=sorted(classes))

    print(f"\nExemple evaluate: {len(y_true)}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    rezultat = {
        "experiment": "Audio-only",
        "n_exemple": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "clase": sorted(classes.tolist()),
        "detalii": detalii,
    }

    with open("rezultate_exp1_audio.json", "w", encoding="utf-8") as f:
        json.dump(rezultat, f, indent=2, ensure_ascii=False)

    print("\nSalvat: rezultate_exp1_audio.json")
    print("=" * 60)


if __name__ == "__main__":
    main()