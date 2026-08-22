import numpy as np
import os

acc_audio = np.load("models/audio_acuratete.npy")
acc_image = np.load("models/imagine_acuratete.npy")
print(f"Acuratete audio: {acc_audio[0]*100:.2f}%")
print(f"Acuratete imagine: {acc_image[0]*100:.2f}%")

print("\nFisiere audio per clasa:")
for cls in sorted(os.listdir("DataSetAudio")):
    path = os.path.join("DataSetAudio", cls)
    if os.path.isdir(path):
        n = len([f for f in os.listdir(path) if f.endswith(".wav")])
        print(f"  {cls}: {n}")