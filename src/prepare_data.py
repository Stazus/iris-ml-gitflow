
"""Przygotowanie danych Iris do trenowania modelu ML."""

from sklearn.model_selection import train_test_split

from load_data import load_iris_data


def prepare_data():
    """Dzieli dane Iris na zbiory treningowy i testowy."""
    df = load_iris_data()

    # X - cechy wykorzystywane przez model
    X = df.drop(columns=["species"])

    # y - gatunek kwiatu, który model ma przewidywać
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    print("=== PODZIAL DANYCH IRIS ===")
    print(f"Zbior treningowy: {X_train.shape}")
    print(f"Zbior testowy: {X_test.shape}")

    print("\nGatunki w zbiorze treningowym:")
    print(y_train.value_counts())

    print("\nGatunki w zbiorze testowym:")
    print(y_test.value_counts())
