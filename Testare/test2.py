import numpy as np

audio_classes = np.load("models/audio_classes.npy", allow_pickle=True)
image_classes = np.load("models/imagine_classes.npy", allow_pickle=True)

print("Audio classes:", sorted(audio_classes))
print("Image classes:", sorted(image_classes))
print("Sunt identice?", sorted(audio_classes) == sorted(image_classes))