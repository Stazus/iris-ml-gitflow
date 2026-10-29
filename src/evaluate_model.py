
"""Ocena skuteczności i zapis bazowego modelu Iris."""

from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from prepare_data import prepare_data
from train_model import train_model


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "iris_logistic_regression.joblib"


def evaluate_model() -> None:
    """Ocenia model na zbiorze testowym i zapisuje go do pliku."""
    _, X_test, _, y_test = prepare_data()

    model, _ = train_model()
    y_pred = model.predict(X_test)

    print("=== OCENA MODELU IRIS ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_test, y_pred))

    print("\n=== MACIERZ POMYLEK ===")
    print("Kolejnosc klas:", ", ".join(model.classes_))
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel zapisano w: {MODEL_PATH}")


if __name__ == "__main__":
    evaluate_model()
