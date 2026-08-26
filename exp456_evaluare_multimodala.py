"""
Experimentele 4, 5, 6 din plan.

Exp 4 - Audio-only pe perechile multimodale
    Trebuie sa stim cat de bine se descurca audio-only pe exact acelasi esntion
    pe care va fi testat si Late Fusion.
    Luam cele 290 de perechi audio-imagine construite la experimentul 3
    si rulam modelul audio doar pe partea audio a fiecarei perechi, ignorand imaginea asociata
    
Exp 5 -Image-only pe perechi multimodale
    Avem nevoie de o valoare de referinta calculata pe acelasi esntion exact, nu pe setul de test propriu al imaginilor
    Folosim exact aceasi logica ca la experimentul 4 dar cu modelul de imagini
    ruleaza doar pe partea de imagine a celor 290 de perechi, ignorand auio-ul
    
Exp 6 - Late Fusion
    Vrem sa demonstram ca folosirea a doua surse de informatie (audio + imagine) e mai robusta decat folosirea uneia singure.
    Daca modelul audio greseste la un anumit fisier dar modelul de imagini nimereste corect (sau invers), combinarea celor doua probabilitati poate trage predictia finala spre rapsunsul corect, chiar daca niciunul din modele nu ar fi fost suficient de sigur singur
    Pentru fiecare din cele 290 de perechi, luam probabilitatile produse de ambele modele (audio si imagine) si le combinam matematic:
        Pfusion(c) = 0.5 x Paudio(c) + 0.5 x Pimagine(c)
    Clasa finala aleasa este cea cu probabilitatea combinata cea mai mare. Aceasta se numeste Late Fusion pentru ca fuziunea are loc dupa ce fiecare model a produs deja predicita sa individuala
"""

import os
import json
import configparser
import numpy as np
import librosa
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read("config.txt")

N_MFCC      = int(config["audio"]["n_mfcc"])
DURATION    = int(config["audio"]["duration"])
SAMPLE_RATE = int(config["audio"]["sample_rate"])
INPUT_SIZE  = int(config["imagine"]["input_size"])
MODELS_DIR  = config["experiment"]["models_experiment_dir"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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


def load_models():
    audio_classes = np.load(os.path.join(MODELS_DIR, "audio_classes.npy"), allow_pickle=True)
    audio_model = AudioClassifier(26, len(audio_classes))
    audio_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "audio_model.pth"), map_location=DEVICE))
    audio_model.to(DEVICE).eval()

    image_classes = np.load(os.path.join(MODELS_DIR, "image_classes.npy"), allow_pickle=True)
    image_model = models.resnet18(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(image_classes))
    image_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "image_model.pth"), map_location=DEVICE))
    image_model.to(DEVICE).eval()

    return audio_model, audio_classes, image_model, image_classes


def predict_audio_probs(model, audio_path):
    y, sr = librosa.load(audio_path, duration=DURATION, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    tensor = torch.tensor(features).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        return torch.softmax(out, dim=1).cpu().numpy()[0]


def predict_image_probs(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        return torch.softmax(out, dim=1).cpu().numpy()[0]


def calculeaza_metrici(y_true, y_pred, clase_ordonate):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=clase_ordonate)
    return acc, prec, rec, f1, cm


def salveaza_matrice_confuzie(cm, clase, titlu, fisier_output):
    fig, ax = plt.subplots(figsize=(max(6, len(clase)*0.6), max(5, len(clase)*0.55)))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(clase)))
    ax.set_yticks(range(len(clase)))
    ax.set_xticklabels(clase, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(clase, fontsize=9)
    ax.set_xlabel("Clasa prezisă")
    ax.set_ylabel("Clasa reală")
    ax.set_title(titlu, fontsize=13, fontweight="bold")

    for i in range(len(clase)):
        for j in range(len(clase)):
            val = cm[i, j]
            culoare = "white" if val > cm.max()/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=culoare, fontsize=8)

    fig.colorbar(im, ax=ax, label="Nr. exemple")
    plt.tight_layout()
    plt.savefig(fisier_output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {fisier_output}")


def main():
    print("=" * 60)
    print("EXPERIMENTELE 4, 5, 6 — Evaluare pe perechi multimodale")
    print("=" * 60)

    with open("test_pairs_multimodal.json", "r", encoding="utf-8") as f:
        perechi = json.load(f)

    with open("clase_comune.json", "r", encoding="utf-8") as f:
        clase_comune_map = json.load(f)

    audio_model, audio_classes, image_model, image_classes = load_models()
    audio_raw_to_canonic = {c["folder_audio"]: c["nume_canonic"] for c in clase_comune_map}
    image_raw_to_canonic = {c["folder_imagine"]: c["nume_canonic"] for c in clase_comune_map}
    canonic_to_audio_idx = {}
    for idx, raw in enumerate(audio_classes):
        if raw in audio_raw_to_canonic:
            canonic_to_audio_idx[audio_raw_to_canonic[raw]] = idx

    canonic_to_image_idx = {}
    for idx, raw in enumerate(image_classes):
        if raw in image_raw_to_canonic:
            canonic_to_image_idx[image_raw_to_canonic[raw]] = idx

    clase_comune = sorted(canonic_to_audio_idx.keys() & canonic_to_image_idx.keys())

    print(f"\nClase comune folosite in fuziune: {len(clase_comune)}")
    for c in clase_comune:
        print(f"  {c}")
    print(f"\nNumar perechi de evaluat: {len(perechi)}")

    y_true = []
    y_pred_audio = []
    y_pred_image = []
    y_pred_fusion = []

    for pereche in perechi:
        adevarat = pereche["instrument"]

        probs_a = predict_audio_probs(audio_model, pereche["audio_path"])
        probs_i = predict_image_probs(image_model, pereche["image_path"])
        idx_pred_a = int(np.argmax(probs_a))
        raw_pred_a = audio_classes[idx_pred_a]
        pred_a = audio_raw_to_canonic.get(raw_pred_a, raw_pred_a)
        idx_pred_i = int(np.argmax(probs_i))
        raw_pred_i = image_classes[idx_pred_i]
        pred_i = image_raw_to_canonic.get(raw_pred_i, raw_pred_i)
        fusion_probs = []
        for c in clase_comune:
            idx_a = canonic_to_audio_idx[c]
            idx_i = canonic_to_image_idx[c]
            p = 0.5 * probs_a[idx_a] + 0.5 * probs_i[idx_i]
            fusion_probs.append(p)
        pred_f = clase_comune[int(np.argmax(fusion_probs))]

        y_true.append(adevarat)
        y_pred_audio.append(pred_a)
        y_pred_image.append(pred_i)
        y_pred_fusion.append(pred_f)

    clase_ordonate = clase_comune

    print("\n" + "─" * 60)
    print("EXPERIMENT 4 — Audio-only (pe perechi multimodale)")
    print("─" * 60)
    acc4, prec4, rec4, f1_4, cm4 = calculeaza_metrici(y_true, y_pred_audio, clase_ordonate)
    print(f"Accuracy:  {acc4*100:.2f}%")
    print(f"Precision: {prec4:.4f}")
    print(f"Recall:    {rec4:.4f}")
    print(f"F1-score:  {f1_4:.4f}")

    print("\n" + "─" * 60)
    print("EXPERIMENT 5 — Image-only (pe perechi multimodale)")
    print("─" * 60)
    acc5, prec5, rec5, f1_5, cm5 = calculeaza_metrici(y_true, y_pred_image, clase_ordonate)
    print(f"Accuracy:  {acc5*100:.2f}%")
    print(f"Precision: {prec5:.4f}")
    print(f"Recall:    {rec5:.4f}")
    print(f"F1-score:  {f1_5:.4f}")

    print("\n" + "─" * 60)
    print("EXPERIMENT 6 — Late Fusion")
    print("─" * 60)
    acc6, prec6, rec6, f1_6, cm6 = calculeaza_metrici(y_true, y_pred_fusion, clase_ordonate)
    print(f"Accuracy:  {acc6*100:.2f}%")
    print(f"Precision: {prec6:.4f}")
    print(f"Recall:    {rec6:.4f}")
    print(f"F1-score:  {f1_6:.4f}")

    print("\nGenerare matrici de confuzie...")
    salveaza_matrice_confuzie(cm4, clase_ordonate, "Matrice de confuzie — Audio-only", "cm_audio_only.png")
    salveaza_matrice_confuzie(cm5, clase_ordonate, "Matrice de confuzie — Image-only", "cm_image_only.png")
    salveaza_matrice_confuzie(cm6, clase_ordonate, "Matrice de confuzie — Late Fusion", "cm_late_fusion.png")

    complementaritate = {"corect_corect": 0, "corect_gresit": 0, "gresit_corect": 0, "gresit_gresit": 0}
    for i in range(len(y_true)):
        a_corect = y_pred_audio[i] == y_true[i]
        i_corect = y_pred_image[i] == y_true[i]
        if a_corect and i_corect:
            complementaritate["corect_corect"] += 1
        elif a_corect and not i_corect:
            complementaritate["corect_gresit"] += 1
        elif not a_corect and i_corect:
            complementaritate["gresit_corect"] += 1
        else:
            complementaritate["gresit_gresit"] += 1

    print("\n" + "─" * 60)
    print("ANALIZA COMPLEMENTARITATE (Audio vs Imagine)")
    print("─" * 60)
    print(f"Ambele corecte:        {complementaritate['corect_corect']}")
    print(f"Doar audio corect:     {complementaritate['corect_gresit']}")
    print(f"Doar imagine corect:   {complementaritate['gresit_corect']}")
    print(f"Ambele gresite:        {complementaritate['gresit_gresit']}")

    rezultat_final = {
        "n_perechi": len(perechi),
        "clase_comune": clase_ordonate,
        "exp4_audio_only": {
            "accuracy": acc4, "precision": prec4, "recall": rec4, "f1_score": f1_4,
            "confusion_matrix": cm4.tolist(),
        },
        "exp5_image_only": {
            "accuracy": acc5, "precision": prec5, "recall": rec5, "f1_score": f1_5,
            "confusion_matrix": cm5.tolist(),
        },
        "exp6_late_fusion": {
            "accuracy": acc6, "precision": prec6, "recall": rec6, "f1_score": f1_6,
            "confusion_matrix": cm6.tolist(),
        },
        "complementaritate": complementaritate,
        "predictii_detaliate": [
            {
                "instrument_real": y_true[i],
                "pred_audio": y_pred_audio[i],
                "pred_image": y_pred_image[i],
                "pred_fusion": y_pred_fusion[i],
            }
            for i in range(len(y_true))
        ],
    }

    with open("rezultate_exp456_final.json", "w", encoding="utf-8") as f:
        json.dump(rezultat_final, f, indent=2, ensure_ascii=False)

    print("\nSalvat: rezultate_exp456_final.json")
    print("=" * 60)


if __name__ == "__main__":
    main()