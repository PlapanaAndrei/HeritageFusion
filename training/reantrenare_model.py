import os
import argparse
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, models, transforms

AUDIO_DATASET_PATH  = os.environ.get("AUDIO_DATASET_PATH",  "DataSetAudio")
IMAGE_DATASET_PATH  = os.environ.get("IMAGE_DATASET_PATH",  "DatasetFinal")
MODELS_DIR          = os.environ.get("MODELS_DIR",          "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def extract_audio_features(file_path: str):
    import librosa
    y, sr = librosa.load(file_path, duration=5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


def reantrenare_audio(epochs: int = 50, log_fn=print) -> float:
    """Reantrenează modelul audio și salvează fișierele .pth/.npy în MODELS_DIR."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    log_fn("[AUDIO] Incarcare dataset...")
    features, labels = [], []

    for label in os.listdir(AUDIO_DATASET_PATH):
        class_path = os.path.join(AUDIO_DATASET_PATH, label)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(".wav"):
                fp = os.path.join(class_path, file)
                try:
                    features.append(extract_audio_features(fp))
                    labels.append(label)
                except Exception as e:
                    log_fn(f"  [SKIP] {fp}: {e}")

    if not features:
        raise RuntimeError("Nu s-au gasit fisiere WAV in AUDIO_DATASET_PATH!")

    features = np.array(features)
    labels   = np.array(labels)

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_enc, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(AudioDataset(X_test,  y_test),  batch_size=32)

    model     = AudioClassifier(13, len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"  [AUDIO] Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            preds = model(X_b).argmax(1)
            total   += y_b.size(0)
            correct += (preds == y_b).sum().item()

    acuratete = correct / total
    log_fn(f"[AUDIO] Acuratete test: {acuratete*100:.2f}%")

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "audio_classes.npy"), le.classes_)
    np.save(os.path.join(MODELS_DIR, "audio_acuratete.npy"),   np.array([acuratete]))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "audio_model.pth"))
    log_fn("[AUDIO] Model salvat cu succes!")
    return acuratete

def reantrenare_imagine(epochs: int = 10, log_fn=print) -> float:
    """Reantrenează modelul de imagini și salvează fișierele .pth/.npy în MODELS_DIR."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    log_fn("[IMAGINE] Incarcare dataset...")
    dataset    = datasets.ImageFolder(IMAGE_DATASET_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    model = models.resnet18(weights="IMAGENET1K_V1")
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"  [IMAGINE] Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            preds = model(X_b).argmax(1)
            total   += y_b.size(0)
            correct += (preds == y_b).sum().item()

    acuratete = correct / total
    log_fn(f"[IMAGINE] Acuratete test: {acuratete*100:.2f}%")

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "imagine_classes.npy"), dataset.classes)
    np.save(os.path.join(MODELS_DIR, "imagine_acuratete.npy"),   np.array([acuratete]))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "image_model.pth"))
    log_fn("[IMAGINE] Model salvat cu succes!")
    return acuratete

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tip",
        choices=["audio", "imagine", "ambele"],
        required=True,
        help="Ce model sa reantrenam",
    )
    parser.add_argument("--epochs-audio",  type=int, default=50)
    parser.add_argument("--epochs-imagine", type=int, default=10)
    args = parser.parse_args()

    if args.tip in ("audio", "ambele"):
        reantrenare_audio(epochs=args.epochs_audio)

    if args.tip in ("imagine", "ambele"):
        reantrenare_imagine(epochs=args.epochs_imagine)