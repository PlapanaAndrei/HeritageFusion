import os
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

config = configparser.ConfigParser()
config.read("config.txt")

DATASET_PATH = config["paths"]["image_dataset"]
MODELS_DIR = config["paths"]["model_dir"]
INPUT_SIZE = int(config["imagine"]["input_size"])
EPOCHS = int(config["imagine"]["epochs"])
LEARNING_RATE = float(config["imagine"]["learning_rate"])
BATCH_SIZE = int(config["imagine"]["batch_size"])
TRAIN_SPLIT = float(config["imagine"]["train_split"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print(f"[IMAGINE] Incarcare dataset din: {DATASET_PATH}")

base_dataset = datasets.ImageFolder(DATASET_PATH, transform= test_transform)
targets = np.array(base_dataset.targets)

print(f"[IMAGINE] Total imagini: {len(base_dataset)}")
print(f"[IMAGINE] Clase detectate: {len(base_dataset.classes)}")

train_idx, test_idx = train_test_split(
    np.arange(len(base_dataset)),
    test_size = 1.0 - TRAIN_SPLIT,
    random_state = 42,
    stratify = targets
)

print(f"[IMAGINE] Train: {len(train_idx)} | Test: {len(test_idx)}")

train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
test_dataset = datasets.ImageFolder (DATASET_PATH, transform=test_transform)

train_ds = Subset(train_dataset, train_idx)
test_ds = Subset(test_dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = True

num_classes = len(base_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"[IMAGINE] Incepe antreanrea: {EPOCHS} epoci, lr={LEARNING_RATE}, batch={BATCH_SIZE}")

best_acc = 0.0
best_state = None

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
    
    scheduler.step()
    
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
    acc = corect / total
    
    print(classification_report(all_labels, all_preds, target_names=base_dataset.classes))
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}"
          f"Acc: {acc*100:.2f}% Best: {best_acc*100:.2f}%"
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    


print(f"[IMAGINE] Acuratete test: {best_acc*100:.2f}%")

os.makedirs(MODELS_DIR, exist_ok=True)
np.save(os.path.join(MODELS_DIR, "imagine_classes.npy"), base_dataset.classes)
np.save(os.path.join(MODELS_DIR, "imagine_acuratete.npy"), np.array([best_acc]))
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "image_model.pth"))
print(f"[IMAGINE] Model salvat in {MODELS_DIR}/")