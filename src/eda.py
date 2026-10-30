
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import get_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = PROJECT_ROOT / "results" / "eda"


def analyze_dataset(dataset_name):
    X, y, categorical_features = get_dataset(dataset_name)

    output_dir = EDA_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save descriptive statistics.
    X.describe().T.to_csv(output_dir / "feature_summary.csv")

    # 2. Save target class counts.
    class_counts = y.value_counts().sort_index()
    class_counts.rename("count").to_csv(
        output_dir / "class_distribution.csv"
    )

    # 3. Plot class distribution.
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(f"{dataset_name}: target class distribution")
    plt.xlabel("Target class")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    # 4. Identify numeric features suitable for distribution plots.
    # Exclude nominal categorical columns, even if stored as numbers.
    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    # Histograms help us inspect skew and unusual distributions.
    if numeric_features:
        X[numeric_features].hist(
            figsize=(14, 10),
            bins=20,
            edgecolor="black",
        )
        plt.suptitle(f"{dataset_name}: feature distributions")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_histograms.png", dpi=150)
        plt.close()

        # Boxplots help flag possible extreme values.
        for col in numeric_features:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=X[col])
            plt.title(f"{dataset_name}: {col}")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"boxplot_{col}.png",
                dpi=150,
            )
            plt.close()

        # Correlation analysis for numeric features.
        # Correlation is descriptive, not proof of causation.
        correlation = X[numeric_features].corr()
        correlation.to_csv(output_dir / "numeric_correlations.csv")

        plt.figure(
            figsize=(
                max(8, len(numeric_features)),
                max(6, len(numeric_features) * 0.7),
            )
        )
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            annot=len(numeric_features) <= 10,
            fmt=".2f",
        )
        plt.title(f"{dataset_name}: numeric feature correlations")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    print(f"\nDataset: {dataset_name}")
    print("Rows:", len(X))
    print("Features:", X.shape[1])
    print("Missing feature values:", int(X.isna().sum().sum()))
    print("Class counts:", class_counts.to_dict())
    print("EDA outputs:", output_dir)


def main():
    for dataset_name in [
        "breast_cancer",
        "heart_disease",
        "diabetes",
    ]:
        analyze_dataset(dataset_name)


if __name__ == "__main__":
    main()
