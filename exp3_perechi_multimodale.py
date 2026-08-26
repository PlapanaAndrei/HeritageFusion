""" 
Experimentul 3 din plan - Construirea setului de test multimodal

Pentru fiecare imagine din test_image, alege un fisier audio din test_audio de acelasi instrument.
class(audio_i) = class(image_i) = y_i

Ambele provin din subsetul 'test' salvat la etapa 2
"""


import json
import random
import configparser

config = configparser.ConfigParser()
config.read("config.txt")

RANDOM_STATE   = int(config["experiment"]["random_state"])
N_PER_CLASA    = int(config["experiment"]["n_perechi_per_clasa"])

random.seed(RANDOM_STATE)


def main():
    print("=" * 60)
    print("EXPERIMENT 3 — Construire perechi multimodale (test)")
    print("=" * 60)

    with open("split_audio.json", "r", encoding="utf-8") as f:
        split_audio = json.load(f)
    with open("split_image.json", "r", encoding="utf-8") as f:
        split_image = json.load(f)
    with open("clase_comune.json", "r", encoding="utf-8") as f:
        clase_comune = json.load(f)

    perechi = []
    for c in clase_comune:
        clasa_audio = c["folder_audio"]
        clasa_imagine = c["folder_imagine"]
        eticheta = c["nume_canonic"]

        test_audio_files = split_audio[clasa_audio]["test"]
        test_image_files = split_image[clasa_imagine]["test"]

        if not test_audio_files or not test_image_files:
            print(f"  [SKIP] {eticheta}: fara fisiere de test intr-o modalitate")
            continue

        n = min(len(test_audio_files), len(test_image_files), N_PER_CLASA)

        audio_sample = random.sample(test_audio_files, n) if len(test_audio_files) >= n else test_audio_files
        image_sample = random.sample(test_image_files, n) if len(test_image_files) >= n else test_image_files

        n_final = min(len(audio_sample), len(image_sample))
        for i in range(n_final):
            perechi.append({
                "instrument": eticheta,
                "audio_path": audio_sample[i],
                "image_path": image_sample[i],
            })

    print(f"\nTotal perechi construite: {len(perechi)}")
    from collections import Counter
    distributie = Counter(p["instrument"] for p in perechi)
    print(f"Clase reprezentate: {len(distributie)}")
    for cls, n in sorted(distributie.items()):
        print(f"  {cls:25s}  {n} perechi")

    with open("test_pairs_multimodal.json", "w", encoding="utf-8") as f:
        json.dump(perechi, f, indent=2, ensure_ascii=False)

    print("\nSalvat: test_pairs_multimodal.json")
    print("=" * 60)


if __name__ == "__main__":
    main()