import numpy as np
import matplotlib.pyplot as plt

classes = np.load("models/audio_classes.npy", allow_pickle=True)
distribution = np.load("models/audio_distribution.npy", allow_pickle=True)

plt.figure(figsize=(12, 8))
plt.pie(distribution, labels=classes, autopct='%1.1f%%')
plt.title("Distribuția claselor în datele de antrenare")
plt.tight_layout()
plt.savefig("distributie_antrenare.png")
plt.show()