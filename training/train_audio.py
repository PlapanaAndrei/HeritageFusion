import os
import configparser
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

config = configparser.ConfigParser()
config.read("config.txt")

DATASET_PATH = config["paths"]["audio_dataset"]
MODELS_DIR = config["paths"]["model_dir"]
N_MFCC = int(config["audio"]["n_mfcc"])
DURATION = int(config["audio"]["duration"])
SAMPLE_RATE = int(config["audio"]["sample_rate"])
EPOCHS = int(config["audio"]["epochs"])
LEARNING_RATE = float(config["audio"]["learning_rate"])
BATCH_SIZE = int(config["audio"]["batch_size"])
TEST_SIZE = float(config["audio"]["test_size"])
RANDOM_STATE = int(config["audio"]["random_state"])
HIDDEN1 = int(config["model"]["audio_hidden1"])
HIDDEN2 = int(config["model"]["audio_hidden2"])
DROPOUT = float(config["model"]["audio_dropout"])

INPUT_SIZE = N_MFCC * 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(HIDDEN2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def extract_features_augumented(file_path: str) -> list:
    y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)
    variants = [y]
    noise = np.random.normal(0, 0.005, len(y))
    variants.append(y + noise)
    try:
        stretched = librosa.effects.time_stretch(y, rate=0.9)
        variants.append(stretched)
    except Exception:
        pass
    results = []
    for y_var in variants:
        mfcc = librosa.feature.mfcc(y=y_var, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        results.append(np.concatenate([mfcc_mean, mfcc_std]))
    return results

print(f"[AUDIO] Incarcare dataset din: {DATASET_PATH}")

raw_counts = {}
for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue
    raw_counts[label] = len([f for f in os.listdir(class_path)
                              if f.lower().endswith(".wav")])

TARGET = int(np.median(list(raw_counts.values())))
print(f"[AUDIO] Target per clasa: {TARGET}")

features, labels = [], []
for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue
    wav_files = [f for f in os.listdir(class_path) if f.lower().endswith(".wav")]
    needs_aug = raw_counts[label] < TARGET
    for file in wav_files:
        file_path = os.path.join(class_path, file)
        try:
            if needs_aug:
                augmented = extract_features_augumented(file_path)
                for feat in augmented:
                    features.append(feat)
                    labels.append(label)
            else:
                features.append(extract_features(file_path))
                labels.append(label)
        except Exception as e:
            print(f" [SKIP] {file_path}: {e}")

features = np.array(features)
labels = np.array(labels)
print(f"[AUDIO] Total fisiere incarcate: {len(features)}")
print(f"[AUDIO] Clase detectate: {sorted(set(labels))}")

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels_encoded
)

train_loader = DataLoader(
    AudioDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    AudioDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(le.classes_)),
    y=y_train
)

class_weight_tensor = torch.tensor(class_weight, dtype=torch.float).to(DEVICE)

class_distribution = Counter(y_train.tolist())
distribution_array = np.array([class_distribution.get(i, 0)
                                for i in range(len(le.classes_))])

model = AudioClassifier(INPUT_SIZE, len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

print(f"[AUDIO] Incepe antrenarea: {EPOCHS} epoci, lr={LEARNING_RATE}, batch={BATCH_SIZE}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(total_loss)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f" Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

model.eval()
all_preds = []
all_labels = []
corect = total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())
        total += y_batch.size(0)
        corect += (preds == y_batch).sum().item()

acuratete = corect / total
print(f"[AUDIO] Acuratete test: {acuratete:.2f}%")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

os.makedirs(MODELS_DIR, exist_ok=True)
np.save(os.path.join(MODELS_DIR, "audio_classes.npy"), le.classes_)
np.save(os.path.join(MODELS_DIR, "audio_acuratete.npy"), np.array([acuratete]))
np.save(os.path.join(MODELS_DIR, "audio_distribution.npy"), distribution_array)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "audio_model.pth"))
print(f"[AUDIO] Model salvat in {MODELS_DIR}/")