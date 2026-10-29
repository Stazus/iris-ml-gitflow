
"""Trenowanie bazowego modelu klasyfikacji Iris."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from prepare_data import prepare_data


def train_model():
    """Trenuje model Logistic Regression na danych Iris."""
    X_train, X_test, y_train, y_test = prepare_data()

    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()

    print("=== TRENOWANIE MODELU IRIS ===")
    print(f"Algorytm: {type(model).__name__}")
    print(f"Liczba klas: {len(model.classes_)}")
    print(f"Gatunki: {', '.join(model.classes_)}")
    print(f"Accuracy: {accuracy:.2%}")
