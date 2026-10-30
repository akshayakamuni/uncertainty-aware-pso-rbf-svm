
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def audit_dataframe(name, df, target_column=None):
    """Display basic data-quality information."""

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values per column:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No pandas-recognized missing values.")
    else:
        print(missing.to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nData types:")
    print(df.dtypes.to_string())

    if target_column is not None and target_column in df.columns:
        print("\nTarget class distribution:")
        print(df[target_column].value_counts(dropna=False).sort_index())

        print("\nTarget proportions:")
        print(
            df[target_column]
            .value_counts(normalize=True, dropna=False)
            .sort_index()
            .round(3)
        )


def audit_breast_cancer():
    """Audit the Wisconsin Diagnostic Breast Cancer dataset."""

    path = DATA_DIR / "breast_cancer" / "wdbc.data"

    if not path.exists():
        print(f"\n[FILE NOT FOUND] {path}")
        return

    columns = ["id", "diagnosis"] + [
        f"feature_{i}" for i in range(1, 31)
    ]

    df = pd.read_csv(path, header=None, names=columns)

    print("\nNote: 'id' is an identifier, not a predictive feature.")
    print("Diagnosis labels: B = benign, M = malignant.")

    audit_dataframe(
        "Breast Cancer Wisconsin Diagnostic",
        df,
        target_column="diagnosis",
    )


def audit_heart_disease():
    """Audit the processed Cleveland Heart Disease dataset."""

    path = DATA_DIR / "heart_disease" / "processed.cleveland.data"

    if not path.exists():
        print(f"\n[FILE NOT FOUND] {path}")
        return

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

    # Missing entries in this file are represented by '?'.
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        na_values="?",
    )

    audit_dataframe(
        "Heart Disease (Cleveland)",
        df,
        target_column="target",
    )

    print("\nOriginal target interpretation:")
    print("0 = no disease; 1-4 = disease presence categories.")
    print("Any binary mapping must be documented before modeling.")


def audit_diabetes():
    """Audit the Diabetes dataset with comment lines in the file."""

    path = DATA_DIR / "diabetes" / "Diabetes.csv"

    if not path.exists():
        print(f"\n[FILE NOT FOUND] {path}")
        return

    # Ignore lines starting with '#'.
    # The visible file header describes 8 predictors and 1 target.
    columns = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "target",
    ]

    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=columns,
        skipinitialspace=True,
    )

    # Convert columns to numeric; malformed entries become NaN
    # so they can be detected by the audit.
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    audit_dataframe(
        "Diabetes",
        df,
        target_column="target",
    )

    # Zero values need domain-specific review; don't replace them
    # automatically during the audit.
    possible_missing_zero_columns = [
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
    ]

    print("\nZero counts in measurements that need review:")
    print(df[possible_missing_zero_columns].eq(0).sum().to_string())

    print("\nNote:")
    print(
        "Some zero measurements may represent missing or invalid "
        "values. Investigate them before preprocessing."
    )
    print("Pregnancies = 0 can be a valid value.")


def main():
    print("MEDICAL DATASET QUALITY AUDIT")
    print(f"Expected raw data directory: {DATA_DIR}")

    audit_breast_cancer()
    audit_heart_disease()
    audit_diabetes()

    print("\n" + "=" * 70)
    print("DATASET AUDIT FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
