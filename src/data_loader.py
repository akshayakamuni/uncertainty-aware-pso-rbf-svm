
from pathlib import Path

import pandas as pd


# Project root and dataset directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_breast_cancer():
    """Load the Wisconsin Diagnostic Breast Cancer dataset."""

    path = DATA_DIR / "breast_cancer" / "wdbc.data"

    columns = (
        ["id", "diagnosis"]
        + [f"feature_{i}" for i in range(1, 31)]
    )

    df = pd.read_csv(path, header=None, names=columns)

    # The ID identifies a sample; it is not a predictive feature.
    X = df.drop(columns=["id", "diagnosis"])

    # M = malignant (1), B = benign (0)
    y = (df["diagnosis"] == "M").astype(int)
    y.name = "target"

    categorical_features = []

    return X, y, categorical_features


def load_heart_disease():
    """Load the Cleveland Heart Disease dataset."""

    path = DATA_DIR / "heart_disease" / "processed.cleveland.data"

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        na_values="?",
        skipinitialspace=True,
    )

    # Convert all columns to numeric values.
    df = df.apply(pd.to_numeric, errors="coerce")

    X = df.drop(columns=["target"])

    # Original target 0 = no disease.
    # Original targets 1, 2, 3, 4 = disease present.
    y = (df["target"] > 0).astype(int)
    y.name = "target"

    # These variables are treated as categorical.
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    return X, y, categorical_features


def load_diabetes():
    """Load the diabetes dataset and handle suspicious zero values."""

    path = DATA_DIR / "diabetes" / "Diabetes.csv"

    columns = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "target",
    ]

    # Ignore comment lines beginning with #.
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=columns,
        skipinitialspace=True,
    )

    # Ensure the columns contain numeric values.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Zero pregnancies is a valid value, so do not change it.
    # Zeros in these medical measurements may represent
    # unrecorded or invalid measurements.
    zero_as_missing = [
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
    ]

    df[zero_as_missing] = df[zero_as_missing].replace(0, float("nan"))

    X = df.drop(columns=["target"])

    y = df["target"].astype(int)
    y.name = "target"

    categorical_features = []

    return X, y, categorical_features


def get_dataset(name):
    """Return features, target labels, and categorical feature names."""

    loaders = {
        "breast_cancer": load_breast_cancer,
        "heart_disease": load_heart_disease,
        "diabetes": load_diabetes,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Choose one of: {', '.join(loaders.keys())}"
        )

    return loaders[name]()


if __name__ == "__main__":
    for dataset_name in [
        "breast_cancer",
        "heart_disease",
        "diabetes",
    ]:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        X, y, categorical_features = get_dataset(dataset_name)

        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
        print("Missing feature values:", int(X.isna().sum().sum()))
        print("Class counts:")
        print(y.value_counts().sort_index())
        print("Categorical features:", categorical_features)
