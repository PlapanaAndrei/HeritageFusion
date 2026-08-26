""" 
Antrenam modelul de imagini (ResNet18) doar pe fisierele de split_image.json
Augmentarea simpla (flip + rotatie) aplicata doar pe train prin transforms
"""


import os
import json
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder

config = configparser.ConfigParser()
config.read("config.txt")

INPUT_SIZE    = int(config["imagine"]["input_size"])
EPOCHS        = int(config["imagine"]["epochs"])
LEARNING_RATE = float(config["imagine"]["learning_rate"])
BATCH_SIZE    = int(config["imagine"]["batch_size"])
MODELS_DIR    = config["experiment"]["models_experiment_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODELS_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transform_val = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def main():
    print("=" * 60)
    print("ANTRENARE MODEL IMAGINE (split experimental)")
    print("=" * 60)

    with open("split_image.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    for clasa, s in split.items():
        for fp in s["train"]:
            train_files.append(fp)
            train_labels.append(clasa)
        for fp in s["val"]:
            val_files.append(fp)
            val_labels.append(clasa)

    print(f"\nTrain: {len(train_files)} imagini")
    print(f"Val:   {len(val_files)} imagini")

    le = LabelEncoder()
    le.fit(sorted(split.keys()))
    train_labels_enc = le.transform(train_labels)
    val_labels_enc   = le.transform(val_labels)

    train_ds = ImageDataset(train_files, train_labels_enc, transform_train)
    val_ds   = ImageDataset(val_files, val_labels_enc, transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = models.resnet18(weights="IMAGENET1K_V1")
    for p in model.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(le.classes_))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"  Epoch {epoch+1}/{EPOCHS}  Loss={total_loss:.4f}  Val_Acc={val_acc*100:.2f}%")

    print(f"\nCea mai buna acuratete pe validation: {best_val_acc*100:.2f}%")

    np.save(os.path.join(MODELS_DIR, "image_classes.npy"), le.classes_)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "image_model.pth"))
    print(f"Model salvat in {MODELS_DIR}/image_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()