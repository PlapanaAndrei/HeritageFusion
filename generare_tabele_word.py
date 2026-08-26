import json


def main():
    with open("rezultate_exp1_audio.json", "r", encoding="utf-8") as f:
        exp1 = json.load(f)
    with open("rezultate_exp2_imagine.json", "r", encoding="utf-8") as f:
        exp2 = json.load(f)
    with open("rezultate_exp456_final.json", "r", encoding="utf-8") as f:
        exp456 = json.load(f)

    linii = []
    linii.append("=" * 70)
    linii.append("RAPORT FINAL — Evaluare multimodala HeritageFusion")
    linii.append("=" * 70)

    linii.append("\n\nTABEL 1 — Rezultate Audio-only (set de test propriu)")
    linii.append("-" * 70)
    linii.append(f"{'Experiment':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<10}")
    linii.append(
        f"{'Audio-only':<15}"
        f"{exp1['accuracy']*100:<11.2f}%"
        f"{exp1['precision']:<12.4f}"
        f"{exp1['recall']:<12.4f}"
        f"{exp1['f1_score']:<10.4f}"
    )
    linii.append(f"(Evaluat pe {exp1['n_exemple']} exemple din setul de test)")

    linii.append("\n\nTABEL 2 — Rezultate Image-only (set de test propriu)")
    linii.append("-" * 70)
    linii.append(f"{'Experiment':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<10}")
    linii.append(
        f"{'Image-only':<15}"
        f"{exp2['accuracy']*100:<11.2f}%"
        f"{exp2['precision']:<12.4f}"
        f"{exp2['recall']:<12.4f}"
        f"{exp2['f1_score']:<10.4f}"
    )
    linii.append(f"(Evaluat pe {exp2['n_exemple']} exemple din setul de test)")

    linii.append("\n\nTABEL PRINCIPAL — Comparatie pe acelasi set de perechi multimodale")
    linii.append(f"({exp456['n_perechi']} perechi audio-imagine, {len(exp456['clase_comune'])} clase comune)")
    linii.append("-" * 70)
    linii.append(f"{'Model':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<10}")

    a = exp456["exp4_audio_only"]
    i = exp456["exp5_image_only"]
    fu = exp456["exp6_late_fusion"]

    linii.append(
        f"{'Audio-only':<15}"
        f"{a['accuracy']*100:<11.2f}%"
        f"{a['precision']:<12.4f}"
        f"{a['recall']:<12.4f}"
        f"{a['f1_score']:<10.4f}"
    )
    linii.append(
        f"{'Image-only':<15}"
        f"{i['accuracy']*100:<11.2f}%"
        f"{i['precision']:<12.4f}"
        f"{i['recall']:<12.4f}"
        f"{i['f1_score']:<10.4f}"
    )
    linii.append(
        f"{'Late Fusion':<15}"
        f"{fu['accuracy']*100:<11.2f}%"
        f"{fu['precision']:<12.4f}"
        f"{fu['recall']:<12.4f}"
        f"{fu['f1_score']:<10.4f}"
    )

    c = exp456["complementaritate"]
    total = sum(c.values())
    linii.append("\n\nTABEL — Analiza complementaritatii modalitatilor")
    linii.append("-" * 70)
    linii.append(f"{'Audio':<10}{'Imagine':<10}{'Nr. cazuri':<12}{'Procent':<10}")
    linii.append(f"{'Corect':<10}{'Corect':<10}{c['corect_corect']:<12}{c['corect_corect']/total*100:<9.1f}%")
    linii.append(f"{'Corect':<10}{'Gresit':<10}{c['corect_gresit']:<12}{c['corect_gresit']/total*100:<9.1f}%")
    linii.append(f"{'Gresit':<10}{'Corect':<10}{c['gresit_corect']:<12}{c['gresit_corect']/total*100:<9.1f}%")
    linii.append(f"{'Gresit':<10}{'Gresit':<10}{c['gresit_gresit']:<12}{c['gresit_gresit']/total*100:<9.1f}%")

    linii.append("\n\nCONCLUZIE AUTOMATA")
    linii.append("-" * 70)
    best_unimodal = max(a["accuracy"], i["accuracy"])
    diferenta = (fu["accuracy"] - best_unimodal) * 100
    if diferenta > 0:
        linii.append(
            f"Late Fusion imbunatateste performanta cu {diferenta:.2f} puncte procentuale "
            f"fata de cel mai bun model unimodal "
            f"({'Audio' if a['accuracy'] > i['accuracy'] else 'Imagine'}, "
            f"{best_unimodal*100:.2f}%)."
        )
    else:
        linii.append(
            f"Late Fusion NU imbunatateste performanta fata de cel mai bun model unimodal "
            f"(diferenta: {diferenta:.2f} puncte procentuale). Verifica ponderile de fuziune "
            f"sau calitatea perechilor construite."
        )

    raport = "\n".join(linii)
    print(raport)

    with open("raport_final.txt", "w", encoding="utf-8") as f:
        f.write(raport)

    print("\n\nSalvat: raport_final.txt")
    print("Copiaza tabelele de mai sus direct in documentul Word.")
    print("Insereaza si cele 3 imagini: cm_audio_only.png, cm_image_only.png, cm_late_fusion.png")


if __name__ == "__main__":
    main()