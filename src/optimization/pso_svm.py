
from pathlib import Path

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


# Reproducible initial experiment settings
DATASET_NAME = "diabetes"
N_SPLITS = 5
RANDOM_STATE = 42

# Small PSO run for testing the workflow
N_PARTICLES = 6
N_ITERATIONS = 4

# Search in log10 space:
# C and gamma range from 10^-3 to 10^3.
LOWER_BOUND = -3.0
UPPER_BOUND = 3.0

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_particle(log_parameters, X, y, numeric_features,
                      categorical_features, splits):
    """
    Evaluate one SVM configuration using identical CV folds.

    The particle contains:
        log10(C), log10(gamma)

    Returns:
        Mean ROC-AUC loss and fold-level losses.
    """

    log_c, log_gamma = log_parameters

    C = 10 ** float(log_c)
    gamma = 10 ** float(log_gamma)

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
                    C=C,
                    gamma=gamma,
                ),
            ),
        ]
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=splits,
        scoring="roc_auc",
        n_jobs=-1,
    )

    # Convert higher-is-better ROC-AUC to lower-is-better loss.
    fold_losses = 1.0 - scores

    return float(fold_losses.mean()), fold_losses


def run_pso(X, y, numeric_features, categorical_features, splits):
    """
    Basic Particle Swarm Optimization minimizing mean inner-CV loss.

    This is a preliminary optimizer. Its personal/global best positions
    are updated using mean loss, not the uncertainty-aware score.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    positions = rng.uniform(
        LOWER_BOUND,
        UPPER_BOUND,
        size=(N_PARTICLES, 2),
    )

    velocities = rng.uniform(
        -0.5,
        0.5,
        size=(N_PARTICLES, 2),
    )

    personal_best_positions = positions.copy()
    personal_best_losses = np.full(N_PARTICLES, np.inf)

    global_best_position = None
    global_best_loss = np.inf

    history = []
    evaluation_id = 0

    # Standard PSO coefficients
    inertia = 0.7
    cognitive = 1.5
    social = 1.5

    for iteration in range(N_ITERATIONS):
        print(f"\nPSO iteration {iteration + 1}/{N_ITERATIONS}")

        for particle_index in range(N_PARTICLES):
            position = positions[particle_index].copy()

            mean_loss, fold_losses = evaluate_particle(
                position,
                X,
                y,
                numeric_features,
                categorical_features,
                splits,
            )

            log_c, log_gamma = position
            C = 10 ** float(log_c)
            gamma = 10 ** float(log_gamma)

            evaluation_id += 1

            history.append(
                {
                    "evaluation_id": evaluation_id,
                    "iteration": iteration + 1,
                    "particle": particle_index,
                    "log10_C": float(log_c),
                    "log10_gamma": float(log_gamma),
                    "C": C,
                    "gamma": gamma,
                    "mean_loss": mean_loss,
                    "std_loss": float(
                        np.std(fold_losses, ddof=1)
                    ),
                    "mean_roc_auc": 1.0 - mean_loss,
                    **{
                        f"fold_{fold + 1}_loss": float(loss)
                        for fold, loss in enumerate(fold_losses)
                    },
                }
            )

            print(
                f"  Particle {particle_index}: "
                f"C={C:.5g}, gamma={gamma:.5g}, "
                f"mean loss={mean_loss:.4f}"
            )

            if mean_loss < personal_best_losses[particle_index]:
                personal_best_losses[particle_index] = mean_loss
                personal_best_positions[particle_index] = position.copy()

            if mean_loss < global_best_loss:
                global_best_loss = mean_loss
                global_best_position = position.copy()

        # Move particles toward their personal and global best positions.
        r1 = rng.random(size=(N_PARTICLES, 2))
        r2 = rng.random(size=(N_PARTICLES, 2))

        velocities = (
            inertia * velocities
            + cognitive * r1
            * (personal_best_positions - positions)
            + social * r2
            * (global_best_position - positions)
        )

        positions = positions + velocities
        positions = np.clip(
            positions,
            LOWER_BOUND,
            UPPER_BOUND,
        )

    return history


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

    # Materialize once: every PSO evaluation uses the same folds.
    splits = list(cv.split(X, y))

    print(f"Dataset: {DATASET_NAME}")
    print(f"Records: {len(X)}")
    print(f"Particles: {N_PARTICLES}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Search evaluations: {N_PARTICLES * N_ITERATIONS}")

    history = run_pso(
        X,
        y,
        numeric_features,
        categorical_features,
        splits,
    )

    history_df = pd.DataFrame(history)

    history_path = OUTPUT_DIR / "pso_diabetes_search_history.csv"
    history_df.to_csv(history_path, index=False)

    # Re-select among all configurations actually evaluated.
    # This compares the two selection rules on the recorded search.
    # It is exploratory, not an unbiased final performance estimate.
    unique_candidates = (
        history_df.sort_values("evaluation_id")
        .drop_duplicates(subset=["log10_C", "log10_gamma"])
        .reset_index(drop=True)
    )

    fold_columns = [
        f"fold_{fold + 1}_loss"
        for fold in range(N_SPLITS)
    ]

    fold_losses = unique_candidates[fold_columns].to_numpy()

    print("\n" + "=" * 60)
    print("SELECTION FROM RECORDED PSO SEARCH")
    print("=" * 60)

    standard = select_standard_candidate(fold_losses)

    standard_candidate = unique_candidates.iloc[
        standard["best_index"]
    ]

    print("\nStandard selection")
    print("C:", standard_candidate["C"])
    print("gamma:", standard_candidate["gamma"])
    print("Mean loss:", standard["mean_loss"])
    print("Mean ROC-AUC:", 1.0 - standard["mean_loss"])

    for lambda_value in LAMBDA_VALUES:
        result = select_uncertainty_aware_candidate(
            fold_losses,
            lambda_value=lambda_value,
        )

        candidate = unique_candidates.iloc[result["best_index"]]

        print(f"\nUncertainty-aware selection, lambda={lambda_value}")
        print("C:", candidate["C"])
        print("gamma:", candidate["gamma"])
        print("Mean loss:", result["mean_loss"])
        print("Loss standard deviation:", result["std_loss"])
        print("Selection score:", result["selection_score"])

    print(f"\nSearch history saved to: {history_path}")


if __name__ == "__main__":
    main()
