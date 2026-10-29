
"""Eksploracyjna analiza danych (EDA) zbioru Iris."""

from load_data import load_iris_data


def analyze_iris() -> None:
    """Wyświetla podstawową analizę statystyczną Iris."""

    df = load_iris_data()

    print("\n=== STATYSTYKI OPISOWE ===")
    print(df.describe().round(2))

    print("\n=== BRAKUJĄCE WARTOŚCI ===")
    print(df.isnull().sum())

    print("\n=== ZDUPLIKOWANE WIERSZE ===")
    print(df.duplicated().sum())

    print("\n=== ŚREDNIE CECH WEDŁUG GATUNKU ===")
    print(df.groupby("species").mean().round(2))

    print("\n=== MACIERZ KORELACJI ===")
    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.corr().round(2))


if __name__ == "__main__":
    analyze_iris()
