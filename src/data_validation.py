
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def check(condition, message):
    """Print a validation result."""
    status = "PASS" if condition else "CHECK"
    print(f"[{status}] {message}")


def validate_breast_cancer():
    print("\n" + "=" * 60)
    print("BREAST CANCER DATASET VALIDATION")
    print("=" * 60)

    path = DATA_DIR / "breast_cancer" / "wdbc.data"

    columns = ["id", "diagnosis"] + [
        f"feature_{i}" for i in range(1, 31)
    ]
    df = pd.read_csv(path, header=None, names=columns)

    check(len(df) == 569, f"Expected 569 rows; found {len(df)}")
    check(df.shape[1] == 32, f"Expected 32 columns; found {df.shape[1]}")
    check(df["id"].is_unique, "Sample IDs are unique")
    check(
        set(df["diagnosis"].unique()) == {"B", "M"},
        "Diagnosis labels are B and M",
    )
    check(not df.isna().any().any(), "No missing values detected")
    check(not df.duplicated().any(), "No duplicate rows detected")

    features = df.drop(columns=["id", "diagnosis"])
    check(
        features.select_dtypes(include="number").shape[1] == 30,
        "Exactly 30 numeric predictors are available",
    )


def validate_heart_disease():
    print("\n" + "=" * 60)
    print("HEART DISEASE DATASET VALIDATION")
    print("=" * 60)

    path = DATA_DIR / "heart_disease" / "processed.cleveland.data"

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target",
    ]

    df = pd.read_csv(
        path, header=None, names=columns, na_values="?"
    )

    check(len(df) == 303, f"Expected 303 rows; found {len(df)}")
    check(df.shape[1] == 14, f"Expected 14 columns; found {df.shape[1]}")
    check(
        set(df["target"].dropna().unique()).issubset({0, 1, 2, 3, 4}),
        "Target values belong to the expected 0-4 range",
    )
    check(
        df["target"].notna().all(),
        "Target column contains no missing values",
    )
    check(
        not df.duplicated().any(),
        "No duplicate rows detected",
    )

    print("\nMissing values by column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string() or "None")

    print("\nProposed binary target mapping:")
    print("Original target 0 -> 0 (no disease)")
    print("Original targets 1-4 -> 1 (disease present)")
    print("This mapping will be applied explicitly during modeling.")


def validate_diabetes():
    print("\n" + "=" * 60)
    print("DIABETES DATASET VALIDATION")
    print("=" * 60)

    path = DATA_DIR / "diabetes" / "Diabetes.csv"

    columns = [
        "pregnancies", "glucose", "blood_pressure",
        "skin_thickness", "insulin", "bmi",
        "diabetes_pedigree_function", "age", "target",
    ]

    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=columns,
        skipinitialspace=True,
    )

    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    check(len(df) == 768, f"Expected 768 rows; found {len(df)}")
    check(df.shape[1] == 9, f"Expected 9 columns; found {df.shape[1]}")
    check(
        set(df["target"].dropna().unique()).issubset({0, 1}),
        "Target values are restricted to 0 and 1",
    )
    check(
        df["target"].notna().all(),
        "Target column contains no missing values",
    )
    check(
        not df.duplicated().any(),
        "No duplicate rows detected",
    )

    print("\nZero counts requiring review:")
    zero_columns = [
        "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi",
    ]
    print(df[zero_columns].eq(0).sum().to_string())

    print("\nImportant: zero counts are not automatically missing values.")
    print("Confirm dataset provenance and document the chosen policy.")


def main():
    print("DATASET VALIDATION")
    print(f"Project data directory: {DATA_DIR}")

    for validator in (
        validate_breast_cancer,
        validate_heart_disease,
        validate_diabetes,
    ):
        try:
            validator()
        except FileNotFoundError as error:
            print(f"\n[ERROR] Dataset file not found: {error.filename}")
        except (ValueError, pd.errors.ParserError) as error:
            print(f"\n[ERROR] Could not validate dataset: {error}")

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
