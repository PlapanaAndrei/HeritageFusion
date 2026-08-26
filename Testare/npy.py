import os
import numpy as np
import matplotlib.pyplot as plt

MODELS_DIR  = "models"        # Folderul cu fișierele .npy
IMAGES_DIR  = "DatasetFinal"  # Folderul cu imaginile

def count_files_per_class(folder_path):
    """Numără fișierele din subfoldere pentru imaginile din DatasetFinal."""
    class_counts = {}
    if not os.path.exists(folder_path):
        print(f"[AVERTISMENT] Folderul {folder_path} nu există!")
        return class_counts

    for item in sorted(os.listdir(folder_path)):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            if len(files) > 0:
                class_counts[item] = len(files)
    return class_counts

def plot_donut_chart(data_dict, title):
    """Generează o diagramă de tip Donut Chart."""
    if not data_dict:
        print(f"[SKIP] Nu există date pentru: {title}")
        return

    labels = list(data_dict.keys())
    counts = list(data_dict.values())

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=labels, 
        autopct='%1.1f%%', 
        pctdistance=0.80, 
        startangle=90,
        textprops=dict(color="black")
    )

    # Inelul interior alb
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)

    for autotext in autotexts:
        autotext.set_fontsize(8)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# --- 1. ÎNCĂRCARE DATE AUDIO DIN FOLDERUL MODELS ---

classes_path = os.path.join(MODELS_DIR, "audio_classes.npy")
dist_path    = os.path.join(MODELS_DIR, "audio_distribution.npy")

if os.path.exists(classes_path) and os.path.exists(dist_path):
    classes = np.load(classes_path, allow_pickle=True)
    distribution = np.load(dist_path, allow_pickle=True)

    # 1. Distribuția Audio ÎNAINTE de augmentare (din .npy)
    data_audio_before = dict(zip(classes, distribution))
    plot_donut_chart(data_audio_before, "Distribuția claselor în DataSetAudio (Înainte de Augmentare)")

    # 2. Distribuția Audio DUPĂ augmentare (Echilibrare la valoarea maximă)
    max_samples = max(distribution)
    data_audio_after = {clasa: max_samples for clasa in classes}
    plot_donut_chart(data_audio_after, "Distribuția claselor în DataSetAudio (După Augmentare / Echilibrare)")
else:
    print(f"[EROARE] Nu s-au găsit fișierele .npy în folderul '{MODELS_DIR}'!")

# --- 3. DISTRIBUȚIA PE IMAGINI (DatasetFinal) ---

data_images = count_files_per_class(IMAGES_DIR)
plot_donut_chart(data_images, "Distribuția claselor în DatasetFinal (Imagini - Neaugmentat)")