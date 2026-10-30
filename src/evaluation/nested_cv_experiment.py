from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_FOLDS = 5
INNER_FOLDS = 5

N_PARTICLES = 6
N_ITERATIONS = 4

LOWER_BOUND = -3.0
UPPER_BOUND = 3.0

LAMBDA_VALUE = 1.0

SVD_COMPONENTS = 5
PCA_VARIANCE = 0.95

# Check these names against src/data_loader.py.
DATASETS = [
    "diabetes",
    "breast_cancer",
    "heart_disease",
]

PREPROCESSING_CONFIGS = [
    {
        "name": "baseline_median",
        "numeric_imputation": "median",
        "dimensionality_reduction": None,
    },
    {
        "name": "mean_imputation",
        "numeric_imputation": "mean",
        "dimensionality_reduction": None,
    },
    {
        "name": "pca_median",
        "numeric_imputation": "median",
        "dimensionality_reduction": "pca",
    },
    {
        "name": "svd_median",
        "numeric_imputation": "median",
        "dimensionality_reduction": "svd",
    },
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CREATE THE PREPROCESSING + SVM PIPELINE
# ============================================================

def create_model(
    C,
    gamma,
    numeric_features,
    categorical_features,
    numeric_imputation="median",
    dimensionality_reduction=None,
):
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_imputation=numeric_imputation,
        dimensionality_reduction=dimensionality_reduction,
        pca_variance=PCA_VARIANCE,
        svd_components=SVD_COMPONENTS,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(C),
                    gamma=float(gamma),
                ),
            ),
        ]
    )


# ============================================================
# 3. EVALUATE ONE PSO PARTICLE
# ============================================================

def evaluate_particle(
    position,
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    inner_splits,
    numeric_imputation,
    dimensionality_reduction,
):
    log_C, log_gamma = position

    C = 10 ** log_C
    gamma = 10 ** log_gamma

    model = create_model(
        C=C,
        gamma=gamma,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_imputation=numeric_imputation,
        dimensionality_reduction=dimensionality_reduction,
    )

    auc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=inner_splits,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )

    fold_losses = 1.0 - auc_scores

    return C, gamma, fold_losses


# ============================================================
# 4. RUN INNER-CV PARTICLE SWARM OPTIMIZATION
# ============================================================

def run_inner_pso(
    X_train,
    y_train,
    numeric_features,
    categorical_features,
    outer_fold,
    numeric_imputation,
    dimensionality_reduction,
):
    rng = np.random.default_rng(RANDOM_STATE + outer_fold)

    positions = rng.uniform(
        LOWER_BOUND,
        UPPER_BOUND,
        size=(N_PARTICLES, 2),
    )

    velocities = rng.uniform(
        -1.0,
        1.0,
        size=(N_PARTICLES, 2),
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    inner_splits = list(inner_cv.split(X_train, y_train))

    personal_best_positions = positions.copy()
    personal_best_losses = np.full(N_PARTICLES, np.inf)

    global_best_position = None
    global_best_loss = np.inf

    history = []

    inertia = 0.7
    cognitive = 1.5
    social = 1.5

    evaluation_number = 0

    for iteration in range(N_ITERATIONS):
        print(
            f"\nOuter fold {outer_fold}/{OUTER_FOLDS}"
            f" - PSO iteration {iteration + 1}/{N_ITERATIONS}"
        )

        for particle in range(N_PARTICLES):
            position = positions[particle].copy()

            C, gamma, fold_losses = evaluate_particle(
                position=position,
                X_train=X_train,
                y_train=y_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                inner_splits=inner_splits,
                numeric_imputation=numeric_imputation,
                dimensionality_reduction=dimensionality_reduction,
            )

            mean_loss = float(np.mean(fold_losses))
            std_loss = float(np.std(fold_losses, ddof=1))

            evaluation_number += 1

            history.append(
                {
                    "outer_fold": outer_fold,
                    "iteration": iteration + 1,
                    "particle": particle,
                    "evaluation": evaluation_number,
                    "C": C,
                    "gamma": gamma,
                    "mean_inner_loss": mean_loss,
                    "std_inner_loss": std_loss,
                    "fold_losses": fold_losses.tolist(),
                }
            )

            print(
                f"Particle {particle}: C={C:.6g}, "
                f"gamma={gamma:.6g}, "
                f"mean loss={mean_loss:.4f}"
            )

            # IMPORTANT:
            # PSO movement still optimizes mean inner loss.
            # UA is applied during final candidate selection.
            if mean_loss < personal_best_losses[particle]:
                personal_best_losses[particle] = mean_loss
                personal_best_positions[particle] = position.copy()

            if mean_loss < global_best_loss:
                global_best_loss = mean_loss
                global_best_position = position.copy()

        for particle in range(N_PARTICLES):
            r1 = rng.random(2)
            r2 = rng.random(2)

            velocities[particle] = (
                inertia * velocities[particle]
                + cognitive
                * r1
                * (
                    personal_best_positions[particle]
                    - positions[particle]
                )
                + social
                * r2
                * (global_best_position - positions[particle])
            )

            positions[particle] += velocities[particle]

            positions[particle] = np.clip(
                positions[particle],
                LOWER_BOUND,
                UPPER_BOUND,
            )

    # Remove duplicate C/gamma pairs.
    unique_candidates = {}

    for item in history:
        key = (
            round(item["C"], 10),
            round(item["gamma"], 10),
        )

        if key not in unique_candidates:
            unique_candidates[key] = item

    candidates = list(unique_candidates.values())

    return candidates, history, len(history)


# ============================================================
# 5. SELECT STANDARD AND UNCERTAINTY-AWARE CANDIDATES
# ============================================================

def select_candidates(candidates):
    if not candidates:
        raise ValueError("PSO did not produce any candidates.")

    standard_candidate = min(
        candidates,
        key=lambda item: item["mean_inner_loss"],
    )

    uncertainty_candidate = min(
        candidates,
        key=lambda item: (
            item["mean_inner_loss"]
            + LAMBDA_VALUE * item["std_inner_loss"]
        ),
    )

    return {
        "Standard": standard_candidate,
        "Uncertainty-aware": uncertainty_candidate,
    }


# ============================================================
# 6. EVALUATE ON THE OUTER TEST FOLD
# ============================================================

def evaluate_outer_test(
    candidate,
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    method_name,
    numeric_imputation,
    dimensionality_reduction,
):
    C = candidate["C"]
    gamma = candidate["gamma"]

    model = create_model(
        C=C,
        gamma=gamma,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_imputation=numeric_imputation,
        dimensionality_reduction=dimensionality_reduction,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)

    predicted_negative_count = int((predictions == 0).sum())
    predicted_positive_count = int((predictions == 1).sum())

    outer_auc = roc_auc_score(y_test, decision_scores)
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    outer_loss = 1.0 - outer_auc

    if method_name == "Standard":
        selection_score = candidate["mean_inner_loss"]
    else:
        selection_score = (
            candidate["mean_inner_loss"]
            + LAMBDA_VALUE * candidate["std_inner_loss"]
        )

    return {
        "method": method_name,
        "C": C,
        "gamma": gamma,
        "inner_mean_loss": candidate["mean_inner_loss"],
        "inner_std_loss": candidate["std_inner_loss"],
        "selection_score": selection_score,
        "outer_roc_auc": outer_auc,
        "outer_loss": outer_loss,
        "overtuning_gap": (
            outer_loss - candidate["mean_inner_loss"]
        ),
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "predicted_negative_count": predicted_negative_count,
        "predicted_positive_count": predicted_positive_count,
        "decision_score_min": float(decision_scores.min()),
        "decision_score_max": float(decision_scores.max()),
    }


# ============================================================
# 7. RUN ONE DATASET + ONE PREPROCESSING CONFIGURATION
# ============================================================

def run_experiment(dataset_name, config):
    config_name = config["name"]
    numeric_imputation = config["numeric_imputation"]
    dimensionality_reduction = config["dimensionality_reduction"]

    print("\n" + "#" * 72)
    print(f"DATASET: {dataset_name}")
    print(f"CONFIGURATION: {config_name}")
    print(f"Imputation: {numeric_imputation}")
    print(f"Dimensionality reduction: {dimensionality_reduction}")
    print("#" * 72)

    X, y, categorical_features = get_dataset(dataset_name)

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    print(f"Records: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Class counts:\n{y.value_counts().to_dict()}")

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_outer_results = []
    all_search_history = []

    for outer_fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"{dataset_name} | {config_name} | "
            f"OUTER FOLD {outer_fold}/{OUTER_FOLDS}"
        )
        print("=" * 60)

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        candidates, history, evaluation_count = run_inner_pso(
            X_train=X_train,
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            outer_fold=outer_fold,
            numeric_imputation=numeric_imputation,
            dimensionality_reduction=dimensionality_reduction,
        )

        for row in history:
            row["dataset"] = dataset_name
            row["preprocessing_config"] = config_name

        all_search_history.extend(history)

        selected_candidates = select_candidates(candidates)

        for method_name, candidate in selected_candidates.items():
            print(f"\nMethod: {method_name}")
            print(f"C: {candidate['C']:.6g}")
            print(f"gamma: {candidate['gamma']:.6g}")
            print(
                f"Mean inner loss: "
                f"{candidate['mean_inner_loss']:.4f}"
            )
            print(
                f"Inner loss SD: "
                f"{candidate['std_inner_loss']:.4f}"
            )

            metrics = evaluate_outer_test(
                candidate=candidate,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                method_name=method_name,
                numeric_imputation=numeric_imputation,
                dimensionality_reduction=dimensionality_reduction,
            )

            result = {
                "dataset": dataset_name,
                "preprocessing_config": config_name,
                "numeric_imputation": numeric_imputation,
                "dimensionality_reduction": (
                    dimensionality_reduction or "none"
                ),
                "outer_fold": outer_fold,
                "lambda": (
                    LAMBDA_VALUE
                    if method_name == "Uncertainty-aware"
                    else 0.0
                ),
                "unique_candidates": len(candidates),
                "particle_evaluations": evaluation_count,
                **metrics,
            }

            all_outer_results.append(result)

            print(f"Outer ROC-AUC: {metrics['outer_roc_auc']:.4f}")
            print(
                f"Balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"F1-score: {metrics['f1_score']:.4f}")

    results_df = pd.DataFrame(all_outer_results)
    history_df = pd.DataFrame(all_search_history)

    results_path = RESULTS_DIR / (
        f"nested_cv_{dataset_name}_{config_name}_outer_results.csv"
    )

    history_path = RESULTS_DIR / (
        f"nested_cv_{dataset_name}_{config_name}_search_history.csv"
    )

    results_df.to_csv(results_path, index=False)
    history_df.to_csv(history_path, index=False)

    metric_columns = [
        "outer_roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "overtuning_gap",
    ]

    print("\nMean outer-fold metrics:")
    print(
        results_df.groupby("method")[metric_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nResults saved to: {results_path}")
    print(f"Search history saved to: {history_path}")

    return results_df, history_df


# ============================================================
# 8. MAIN: RUN ALL DATASETS AUTOMATICALLY
# ============================================================

def main():
    all_results = []
    completed_datasets = []
    failed_datasets = []

    print("=" * 72)
    print("MULTI-DATASET NESTED CROSS-VALIDATION EXPERIMENT")
    print("=" * 72)
    print(f"Datasets: {DATASETS}")
    print(f"Outer folds: {OUTER_FOLDS}")
    print(f"Inner folds: {INNER_FOLDS}")
    print(f"Particles: {N_PARTICLES}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Lambda: {LAMBDA_VALUE}")
    print(f"Configurations per dataset: {len(PREPROCESSING_CONFIGS)}")

    for dataset_name in DATASETS:
        print("\n" + "=" * 72)
        print(f"STARTING DATASET: {dataset_name}")
        print("=" * 72)

        try:
            # Validate dataset name before running the expensive search.
            X, y, _ = get_dataset(dataset_name)

            if y.nunique() != 2:
                raise ValueError(
                    f"{dataset_name} must have exactly two target classes."
                )

            dataset_results = []

            for config in PREPROCESSING_CONFIGS:
                results_df, _ = run_experiment(
                    dataset_name,
                    config,
                )
                dataset_results.append(results_df)

            combined = pd.concat(
                dataset_results,
                ignore_index=True,
            )

            combined_path = RESULTS_DIR / (
                f"nested_cv_{dataset_name}_preprocessing_comparison.csv"
            )
            combined.to_csv(combined_path, index=False)

            metric_columns = [
                "outer_roc_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1_score",
                "overtuning_gap",
            ]

            summary = (
                combined.groupby(
                    ["preprocessing_config", "method"]
                )[metric_columns]
                .mean()
                .round(4)
            )

            summary_path = RESULTS_DIR / (
                f"nested_cv_{dataset_name}_preprocessing_summary.csv"
            )
            summary.to_csv(summary_path)

            all_results.append(combined)
            completed_datasets.append(dataset_name)

            print(f"\nCompleted dataset: {dataset_name}")
            print(summary.to_string())
            print(f"Saved: {combined_path}")
            print(f"Saved: {summary_path}")

        except Exception as exc:
            failed_datasets.append(
                {
                    "dataset": dataset_name,
                    "error": str(exc),
                }
            )

            print(f"\nDATASET FAILED: {dataset_name}")
            print(f"Reason: {exc}")
            print(
                "Continuing to the next dataset. "
                "Check this dataset name or loader if needed."
            )

    # Cross-dataset combined results.
    if all_results:
        all_datasets_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        all_datasets_path = RESULTS_DIR / (
            "nested_cv_all_datasets_preprocessing_comparison.csv"
        )
        all_datasets_df.to_csv(
            all_datasets_path,
            index=False,
        )

        metric_columns = [
            "outer_roc_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
            "overtuning_gap",
        ]

        cross_dataset_summary = (
            all_datasets_df.groupby(
                ["dataset", "preprocessing_config", "method"]
            )[metric_columns]
            .mean()
            .round(4)
        )

        cross_dataset_path = RESULTS_DIR / (
            "nested_cv_all_datasets_preprocessing_summary.csv"
        )
        cross_dataset_summary.to_csv(cross_dataset_path)

        print("\n" + "=" * 72)
        print("CROSS-DATASET EXPERIMENT SUMMARY")
        print("=" * 72)
        print(cross_dataset_summary.to_string())
        print(f"\nCombined results: {all_datasets_path}")
        print(f"Combined summary: {cross_dataset_path}")

    print("\nCompleted datasets:", completed_datasets)
    print("Failed datasets:", failed_datasets)


if __name__ == "__main__":
    main()