import matplotlib.pyplot as plt
import os

def get_class_counts(dataset_dir):
    counts = {}
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        file_count = 0
        for root, _, files in os.walk(class_path):
            for filename in files:
                if filename.startswith('.'):
                    continue
                file_count += 1
        if file_count > 0:
            counts[class_name] = file_count
    return counts


def plot_class_distribution_pie(counts, title=None, save_path=None):

    labels = list(counts.keys())
    sizes = list(counts.values())

    plt.figure(figsize=(10, 10))
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
        labeldistance=1.05,
    )
    plt.title(title or "Class Distribution")
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DataSetAudio"))
    #dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DatasetFinal"))

    counts = get_class_counts(dataset_path)
    print("Clase găsite:")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    plot_class_distribution_pie(
        counts,
        title=f"Distribuția claselor în {os.path.basename(dataset_path)}",
        save_path=os.path.join(os.path.dirname(__file__), "class_distribution.png"),
    )


if __name__ == "__main__":
    main()

