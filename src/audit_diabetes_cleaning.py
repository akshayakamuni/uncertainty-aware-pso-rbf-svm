
from pathlib import Path

import pandas as pd

from src.data_loader import DATA_DIR, get_dataset


def main():
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

    raw = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=columns,
        skipinitialspace=True,
    )

    raw = raw.apply(pd.to_numeric, errors="coerce")

    X_clean, y, _ = get_dataset("diabetes")

    measurement_columns = [
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
    ]

    rows = []

    for column in columns[:-1]:
        original_missing = int(raw[column].isna().sum())
        original_zeros = int((raw[column] == 0).sum())

        cleaned_missing = int(X_clean[column].isna().sum())

        rows.append({
            "feature": column,
            "raw_missing": original_missing,
            "raw_zeros": original_zeros,
            "cleaned_missing": cleaned_missing,
            "zeros_converted_to_missing": (
                int(
                    (raw[column] == 0).sum()
                    - (X_clean[column].eq(0).sum())
                )
                if column in measurement_columns
                else 0
            ),
        })

    report = pd.DataFrame(rows)

    output_dir = Path("results") / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "diabetes_cleaning_audit.csv"
    report.to_csv(output_path, index=False)

    print(report.to_string(index=False))
    print("\nRows:", len(raw))
    print("Target labels unchanged:", raw["target"].equals(y))
    print("\nAudit saved to:", output_path)


if __name__ == "__main__":
    main()
