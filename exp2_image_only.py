""" 
Experimentul 2 din plan - Clasificare Image-only
Evaluam modelul ResNet18 pe subsetul TEST din split_image.json
"""


import os
import json
import configparser
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

config = configparser.ConfigParser()
config.read("config.txt")

INPUT_SIZE = int(config["imagine"]["input_size"])
MODELS_DIR = config["experiment"]["models_experiment_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def main():
    print("=" * 60)
    print("EXPERIMENT 2 — Image-only (pe setul de test)")
    print("=" * 60)

    classes = np.load(os.path.join(MODELS_DIR, "image_classes.npy"), allow_pickle=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "image_model.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()

    with open("split_image.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    y_true, y_pred = [], []
    detalii = []

    for clasa, s in split.items():
        for fp in s["test"]:
            try:
                img = Image.open(fp).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(DEVICE)
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
        "experiment": "Image-only",
        "n_exemple": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "clase": sorted(classes.tolist()),
        "detalii": detalii,
    }

    with open("rezultate_exp2_imagine.json", "w", encoding="utf-8") as f:
        json.dump(rezultat, f, indent=2, ensure_ascii=False)

    print("\nSalvat: rezultate_exp2_imagine.json")
    print("=" * 60)


if __name__ == "__main__":
    main()