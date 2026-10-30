
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor


def run_baseline(dataset_name):
    """Evaluate a fixed RBF-SVM using five-fold cross-validation."""

    print("\n" + "=" * 65)
    print(f"BASELINE SVM: {dataset_name}")
    print("=" * 65)

    # 1. Load the dataset.
    X, y, categorical_features = get_dataset(dataset_name)

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    # 2. Create preprocessing.
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # 3. Use an ordinary RBF-SVM with fixed parameters.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                ),
            ),
        ]
    )

    # 4. Preserve class proportions in each fold.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # 5. Define evaluation metrics.
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "sensitivity": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }

    # 6. Evaluate the entire pipeline in cross-validation.
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )

    # 7. Report mean and standard deviation across folds.
    rows = []

    for metric in scoring:
        values = results[f"test_{metric}"]

        rows.append(
            {
                "Metric": metric,
                "Mean": values.mean(),
                "Std": values.std(ddof=1),
            }
        )

    summary = pd.DataFrame(rows)

    print("\nFixed parameters: RBF kernel, C=1.0, gamma='scale'")
    print("Cross-validation: 5-fold stratified")
    print("Training-fold preprocessing: enabled")
    print("\nResults (mean and standard deviation):")
    print(
        summary.to_string(
            index=False,
            formatters={
                "Mean": "{:.4f}".format,
                "Std": "{:.4f}".format,
            },
        )
    )

    return summary


def main():
    datasets = [
        "breast_cancer",
        "heart_disease",
        "diabetes",
    ]

    all_results = []

    for dataset_name in datasets:
        summary = run_baseline(dataset_name)
        summary.insert(0, "Dataset", dataset_name)
        all_results.append(summary)

    combined = pd.concat(all_results, ignore_index=True)

    output_dir = "results"
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "baseline_svm_results.csv",
    )

    combined.to_csv(output_path, index=False)

    print(f"\nBaseline results saved to: {output_path}")


if __name__ == "__main__":
    main()
