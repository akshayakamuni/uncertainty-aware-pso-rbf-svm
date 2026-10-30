
from pathlib import Path

import pandas as pd

from src.data_loader import get_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "results" / "data_quality"


def create_data_quality_report(dataset_name):
    """
    Inspect a dataset without modifying or deleting samples.

    Reports:
    - Dataset dimensions
    - Missing values
    - Duplicate feature rows
    - Data types
    - Numeric summaries
    - Target class counts
    """

    X, y, categorical_features = get_dataset(dataset_name)

    # Combine features and target for inspection only.
    df = X.copy()
    df["target"] = y

    # Missing-value counts by column.
    missing_counts = df.isna().sum()

    # Duplicate predictor rows, excluding the target.
    # These are potential duplicates, not automatically errors.
    duplicate_feature_rows = int(X.duplicated().sum())

    # Numeric descriptive statistics.
    numeric_summary = X.describe(include="all").transpose()

    # Data types.
    data_types = X.dtypes.astype(str)

    # Target distribution.
    class_counts = y.value_counts(dropna=False).sort_index()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    missing_counts.rename("missing_count").to_csv(
        REPORT_DIR / f"{dataset_name}_missing_values.csv"
    )

    numeric_summary.to_csv(
        REPORT_DIR / f"{dataset_name}_feature_summary.csv"
    )

    data_types.rename("data_type").to_csv(
        REPORT_DIR / f"{dataset_name}_data_types.csv"
    )

    class_counts.rename("sample_count").to_csv(
        REPORT_DIR / f"{dataset_name}_class_counts.csv"
    )

    report = {
        "dataset": dataset_name,
        "rows": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "missing_feature_values": int(X.isna().sum().sum()),
        "duplicate_feature_rows": duplicate_feature_rows,
        "categorical_features": ", ".join(categorical_features),
        "class_counts": class_counts.to_dict(),
    }

    return report


def main():
    dataset_names = [
        "breast_cancer",
        "heart_disease",
        "diabetes",
    ]

    reports = []

    for dataset_name in dataset_names:
        print("\n" + "=" * 60)
        print(f"DATA QUALITY REPORT: {dataset_name}")
        print("=" * 60)

        report = create_data_quality_report(dataset_name)
        reports.append(report)

        for key, value in report.items():
            print(f"{key}: {value}")

    summary = pd.DataFrame(reports)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        REPORT_DIR / "dataset_quality_summary.csv",
        index=False,
    )

    print("\nSummary saved to:")
    print(REPORT_DIR / "dataset_quality_summary.csv")


if __name__ == "__main__":
    main()
