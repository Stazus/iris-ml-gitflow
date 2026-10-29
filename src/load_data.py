
"""Wczytywanie zbioru Iris na potrzeby analizy EDA."""

from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data() -> pd.DataFrame:
    """Zwraca zbiór Iris jako DataFrame z nazwami gatunków."""
    iris = load_iris(as_frame=True)

    df = iris.data.copy()
    df["species"] = iris.target.map(
        dict(enumerate(iris.target_names))
    )

    return df


if __name__ == "__main__":
    df = load_iris_data()

    print("Pierwsze 5 wierszy:")
    print(df.head())

    print("\nWymiary zbioru:")
    print(df.shape)

    print("\nLiczba kwiatów według gatunku:")
    print(df["species"].value_counts())
