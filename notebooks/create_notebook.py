
"""Tworzenie notebooka Jupyter do analizy zbioru Iris."""

from pathlib import Path

import nbformat as nbf


notebook = nbf.v4.new_notebook()

notebook.cells = [
    nbf.v4.new_markdown_cell(
        "# Iris – Exploratory Data Analysis (EDA)\n\n"
        "Analiza zbioru Iris w ramach projektu Machine Learning."
    ),

    nbf.v4.new_markdown_cell(
        "## 1. Wczytywanie danych\n\n"
        "Zbiór zawiera 150 obserwacji i trzy gatunki irysów."
    ),

    nbf.v4.new_code_cell(
        "from sklearn.datasets import load_iris\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "iris = load_iris(as_frame=True)\n"
        "df = iris.data.copy()\n"
        "df['species'] = iris.target.map(\n"
        "    dict(enumerate(iris.target_names))\n"
        ")\n"
        "df.head()"
    ),

    nbf.v4.new_markdown_cell(
        "## 2. Statystyki opisowe"
    ),

    nbf.v4.new_code_cell(
        "df.describe()"
    ),

    nbf.v4.new_markdown_cell(
        "## 3. Kontrola jakości danych"
    ),

    nbf.v4.new_code_cell(
        "print('Brakujące wartości:')\n"
        "print(df.isnull().sum())\n\n"
        "print('\\nDuplikaty:', df.duplicated().sum())"
    ),

    nbf.v4.new_markdown_cell(
        "## 4. Porównanie gatunków"
    ),

    nbf.v4.new_code_cell(
        "df.groupby('species').mean().round(2)"
    ),

    nbf.v4.new_markdown_cell(
        "## 5. Wizualizacja cech"
    ),

    nbf.v4.new_code_cell(
        "sns.scatterplot(\n"
        "    data=df,\n"
        "    x='petal length (cm)',\n"
        "    y='petal width (cm)',\n"
        "    hue='species'\n"
        ")\n"
        "plt.title('Porownanie gatunkow Iris')\n"
        "plt.show()"
    ),

    nbf.v4.new_markdown_cell(
        "## 6. Macierz korelacji"
    ),

    nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 6))\n"
        "sns.heatmap(\n"
        "    df.select_dtypes(include='number').corr(),\n"
        "    annot=True,\n"
        "    cmap='coolwarm'\n"
        ")\n"
        "plt.show()"
    ),

    nbf.v4.new_markdown_cell(
        "## 7. Wnioski\n\n"
        "- Zbiór nie zawiera brakujących wartości.\n"
        "- Występuje jeden zduplikowany wiersz.\n"
        "- Długość i szerokość płatków są silnie skorelowane.\n"
        "- Cechy płatków pomagają odróżniać gatunki."
    ),
]

notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

output_path = Path(__file__).resolve().parent / "iris_eda.ipynb"

nbf.write(notebook, output_path)

print(f"Notebook utworzony: {output_path}")
print(f"Liczba komórek: {len(notebook.cells)}")
