""" 
Impartim datele originale (audio si imagine) in Train / Validation / Test
doar pentru clasele comune identificate in etapa anterioara

Split-ul se face pe fisierele originale, inainte de orice augmentare.
Augmentarea se aplica doar pe train, in scriptul de antrenare, niciodata aici. 
Astfel se evita data leakage. 
"""



import os 
import json
import random 
import configparser

config = configparser.ConfigParser()
config.read("config.txt")

AUDIO_DIR = config["paths"]["audio_dataset"]
IMAGE_DIR = config["paths"]["image_dataset"]
TRAIN_RATIO = float(config["experiment"]["train_ratio"])
VAL_RATIO = float(config["experiment"]["val_ratio"])
TEST_RATIO = float(config["experiment"]["test_ratio"])
RANDOM_STATE = int(config["experiment"]["random_state"])

random.seed(RANDOM_STATE)

def split_lista(fisiere: list) -> dict:
    fisiere = fisiere.copy()
    random.shuffle(fisiere)
    
    n = len(fisiere)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    return {
        "train": fisiere[:n_train],
        "val": fisiere[n_train:n_train + n_val],
        "test": fisiere[n_train + n_val:],
    }
    
def split_audio(clase_comune: list) -> dict:
    rezultat = {}
    for c in clase_comune:
        clasa = c["folder_audio"]
        folder = os.path.join(AUDIO_DIR, clasa)
        fisiere = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]
        rezultat[clasa] = split_lista(fisiere)
    return rezultat

def split_imagine(clase_comune: list) -> dict:
    rezultat = {}
    for c in clase_comune:
        clasa = c["folder_imagine"]
        folder = os.path.join(IMAGE_DIR, clasa)
        fisiere = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rezultat[clasa] = split_lista(fisiere)
    return rezultat

def main():
    print("=" * 60)
    print("ETAPA 2 - Split Train / Validare/ Test")
    print("=" * 60)
    
    with open("clase_comune.json", "r", encoding="utf-8") as f:
        clase_comune = json.load(f)
        
    print(f"\Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_STATE}")
    
    split_a = split_audio(clase_comune)
    with open("split_audio.json", "w", encoding="utf-8") as f:
        json.dump(split_a, f, indent =2, ensure_ascii=False)
        
    print(f"\n[AUDIO] Split salvat in split_audio.json")
    total_train_a = total_val_a = total_test_a = 0
    for clasa , s in split_a.items():
        print(f"{clasa:25s} train={len(s['train']):4d} val={len(s['val']):3d} test={len(s['test']):3d}")
        total_train_a += len(s["train"])
        total_val_a += len(s["val"])
        total_test_a += len(s["test"])
    print(f"{'TOTAL':25s} train = {total_train_a:4d} val = {total_val_a:3d} test = {total_test_a:3d}")
    
    split_i = split_imagine(clase_comune)
    with open("split_image.json", "w", encoding="utf-8") as f:
        json.dump(split_i, f, indent=2, ensure_ascii=False)
        
    print(f"\n[IMAGINE] split salvat in split_image.json")
    total_train_i = total_val_i = total_test_i = 0 
    for clasa, s in split_i.items():
        print(f"{clasa:25s} train={len(s['train']):4d} val = {len(s['val']):3d} test={len(s['test']):3d}")
        total_train_i += len(s["train"])
        total_val_i += len(s["val"])
        total_test_i += len(s["test"])
    print(f"{'TOTAL':25s} train={total_train_i:4d} val={total_val_i:3d} test={total_test_i:3d}")
    
if __name__ == "__main__":
    main()
        
    
        