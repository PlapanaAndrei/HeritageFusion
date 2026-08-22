import numpy as np

clase_audio = np.load("models/audio_classes.npy", allow_pickle=True)
print("Clase audio:", clase_audio)

clase_imagine = np.load("models/imagine_classes.npy", allow_pickle=True)
print("Clase imagine:", clase_imagine)

acc_audio = np.load("models/audio_acuratete.npy")
print(f"Acuratete audio: {acc_audio[0]*100:.2f}%")

acc_imagine = np.load("models/imagine_acuratete.npy")
print(f"Acuratete imagine: {acc_imagine[0]*100:.2f}%")