
"""Wizualizacja danych zbioru Iris."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_iris_data


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def create_visualizations() -> None:
    """Generuje wykresy eksploracyjnej analizy danych Iris."""

    df = load_iris_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Histogramy cech
    axes = df.hist(figsize=(12, 8), bins=15)
    plt.suptitle("Rozklady cech zbioru Iris")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_histograms.png", dpi=150)
    plt.close()

    # 2. Wykres punktowy
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        s=70,
    )
    plt.title("Dlugosc i szerokosc platkow Iris")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_scatter.png", dpi=150)
    plt.close()

    # 3. Mapa korelacji
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Macierz korelacji cech Iris")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iris_correlation.png", dpi=150)
    plt.close()

    print(f"Wykresy zapisano w: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {path.name}")


if __name__ == "__main__":
    create_visualizations()
