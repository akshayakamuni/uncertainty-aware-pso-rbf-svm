
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor
from src.optimization.selection import (
    LAMBDA_VALUES,
    select_standard_candidate,
    select_uncertainty_aware_candidate,
)


# Dataset for this initial experiment
DATASET_NAME = "diabetes"

# Fixed SVM candidates for testing the selection workflow.
# These are not PSO results.
CANDIDATES = [
    {"C": 0.1, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 1.0, "gamma": 0.01},
    {"C": 10.0, "gamma": 0.01},
]

N_SPLITS = 5
RANDOM_STATE = 42


def main():
    X, y, categorical_features = get_dataset(DATASET_NAME)

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Create the folds once so every candidate uses identical splits.
    splits = list(cv.split(X, y))

    all_fold_losses = []

    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of records: {len(X)}")
    print(f"Number of candidates: {len(CANDIDATES)}")
    print(f"Number of folds: {N_SPLITS}")
    print("\nEvaluating candidates...\n")

    for candidate_index, params in enumerate(CANDIDATES):
        model = Pipeline(
            steps=[
                (
                    "preprocessing",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=params["C"],
                        gamma=params["gamma"],
                    ),
                ),
            ]
        )

        # ROC-AUC is a score where higher is better.
        # Convert it into loss so lower is better.
        scores = cross_val_score(
            model,
            X,
            y,
            cv=splits,
            scoring="roc_auc",
            n_jobs=-1,
        )

        fold_losses = 1.0 - scores
        all_fold_losses.append(fold_losses)

        print(
            f"Candidate {candidate_index}: "
            f"C={params['C']}, gamma={params['gamma']}"
        )
        print("  ROC-AUC:", np.round(scores, 4))
        print(f"  Mean loss: {fold_losses.mean():.4f}")
        print(f"  Loss standard deviation: {fold_losses.std(ddof=1):.4f}")

    # Rows are candidates; columns are the five fold losses.
    all_fold_losses = np.asarray(all_fold_losses)

    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    standard = select_standard_candidate(all_fold_losses)

    print("\nStandard selection:")
    print(standard)
    print("Selected parameters:", CANDIDATES[standard["best_index"]])

    for lambda_value in LAMBDA_VALUES:
        result = select_uncertainty_aware_candidate(
            all_fold_losses,
            lambda_value=lambda_value,
        )

        print(f"\nUncertainty-aware selection, lambda={lambda_value}:")
        print(result)
        print("Selected parameters:", CANDIDATES[result["best_index"]])

    # Save candidate-level results for later analysis.
    rows = []

    for i, params in enumerate(CANDIDATES):
        losses = all_fold_losses[i]

        row = {
            "dataset": DATASET_NAME,
            "candidate_index": i,
            "C": params["C"],
            "gamma": params["gamma"],
            "mean_loss": float(losses.mean()),
            "std_loss": float(losses.std(ddof=1)),
            "mean_roc_auc": float(1.0 - losses.mean()),
        }

        for fold_index, loss in enumerate(losses, start=1):
            row[f"fold_{fold_index}_loss"] = float(loss)

        rows.append(row)

    output_path = (
        "results/inner_cv_diabetes_candidates.csv"
    )

    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nCandidate results saved to: {output_path}")


if __name__ == "__main__":
    main()
