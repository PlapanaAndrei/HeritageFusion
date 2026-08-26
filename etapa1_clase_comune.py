""" 
Identificam intersectia claselor dintre DataSetAudio si DatasetFinal,
normalizand numele acestora ca sa prinda variatii de tip "Acoustic_Guitar" vs "acoustic guitar" vs "AcousticGuitar"
"""

import os 
import json
import configparser 

config = configparser.ConfigParser()
config.read("config.txt")

AUDIO_DIR = config["paths"]["audio_dataset"]
IMAGE_DIR = config["paths"]["image_dataset"]

def normalizeaza(nume: str) -> str:
    return nume.lower().replace("_","").replace("-","").replace(" ", "")

def get_classes_map(path: str) -> dict:
    rezultat= {}
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            rezultat[normalizeaza(d)] = d 
    return rezultat

def main():
    print("=" * 60)
    print("ETAPA 1 - Identificarea clase comune")
    print("=" * 60)
    
    audio_map = get_classes_map(AUDIO_DIR)
    image_map = get_classes_map(IMAGE_DIR)
    
    print(f"\nClase audio gasite: {len(audio_map)}")
    print(f"Clase imagine gasite: {len(image_map)}")
    
    comune_norm = set(audio_map.keys()) & set(image_map.keys())
    doar_audio = set((audio_map.keys())) - set(image_map.keys())
    doar_imagine = set(image_map.keys()) - set(audio_map.keys())
    
    clase_comune = []
    for norm in sorted(comune_norm):
        clase_comune.append({
            "nume_canonic": audio_map[norm],
            "folder_audio": audio_map[norm],
            "folder_imagine": image_map[norm],
        })
    
    print(f"\n{'-'*60}")
    print(f"Clase comune gasite: {len(clase_comune)}")
    print(f"{'-'*60}")
    for c in clase_comune:
        eticheta = c["folder_audio"]
        if c["folder_audio"] != c["folder_imagine"]:
            eticheta += f"(imagine: {c['folder_imagine']})"
        print(f" ✅ {eticheta}")
        
    if doar_audio:
        print(f"\nClase doar in audio ({len(doar_audio)}) - excluse din experiment:")
        for n in sorted(doar_audio):
            print(f" ❌ {audio_map[n]}")
            
    if doar_imagine:
        print(f"\nClase comune in imagine ({len(doar_imagine)}) - excluse din experiment:")
        for n in sorted(doar_imagine):
            print(f" ❌ {image_map[n]}")
            
    if len(clase_comune) < 2:
        print("\nPrea putine clase comune gasite")
        
    with open("clase_comune.json", "w", encoding="utf-8") as f:
        json.dump(clase_comune, f, indent = 2, ensure_ascii=False)
        
    print(f"\nSalvat: clase_comune.json ({len(clase_comune)})")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
            
