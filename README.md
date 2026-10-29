
# Iris Classification – Machine Learning with Git Flow

Projekt klasyfikacji gatunków irysów, realizowany w ramach kursu Data Scientist w LearnIT.

## Cel projektu

Celem projektu jest zbudowanie modelu Machine Learning, który rozpoznaje gatunek irysa na podstawie czterech cech:

- Długość działki kielicha (sepal length)
- Szerokość działki kielicha (sepal width)
- Długość płatka (petal length)
- Szerokość płatka (petal width)

Model będzie klasyfikował kwiaty do trzech gatunków:
- Iris setosa
- Iris versicolor
- Iris virginica

## Technologie

- Python 3.12
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook
- Git i GitHub

## Struktura projektu

```text
iris-ml-gitflow/
├── data/           # Dane projektu (ignorowane przez Git)
├── notebooks/      # Eksploracyjna analiza danych (EDA)
├── src/            # Kod źródłowy Machine Learning
├── models/         # Wytrenowane modele (ignorowane przez Git)
├── .gitignore
├── README.md
└── requirements.txt
```

## Etapy realizacji

1. Przygotowanie struktury repozytorium.
2. Eksploracyjna analiza danych (EDA).
3. Przygotowanie danych treningowych i testowych.
4. Trenowanie bazowego modelu klasyfikacyjnego.
5. Ocena modelu za pomocą accuracy i macierzy pomyłek.
6. Code Review i Pull Requests.
7. Wydanie wersji v1.0.

## Git Flow

Projekt wykorzystuje następujące gałęzie:

- `main` – stabilna wersja projektu.
- `develop` – integracja nowych funkcjonalności.
- `feature-eda` – eksploracyjna analiza danych.
- `feature-baseline-model` – implementacja modelu bazowego.

Zmiany z gałęzi funkcjonalnych będą scalane przez Pull Requests.

## Status

Projekt w trakcie realizacji.

## Autor

Stazus – GitHub: https://github.com/Stazus
