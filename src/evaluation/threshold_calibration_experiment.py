"""Leakage-safe threshold calibration across multiple datasets.

For each dataset, preprocessing configuration and outer fold:
1. Run the existing PSO search using inner CV.
2. Select Standard and Uncertainty-aware candidates.
3. Obtain inner out-of-fold decision scores.
4. Choose the threshold maximizing inner OOF balanced accuracy.
5. Refit on outer-training data and evaluate on the untouched outer test fold.

Outer-test labels are never used to choose the threshold.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.evaluation.nested_cv_experiment import (
    RANDOM_STATE,
    OUTER_FOLDS,
    INNER_FOLDS,
    LAMBDA_VALUE,
    PREPROCESSING_CONFIGS,
    RESULTS_DIR,
    create_model,
    run_inner_pso,
    select_candidates,
)
from src.data_loader import get_dataset


# Keep the names consistent with src/data_loader.py.
DATASETS = [
    "diabetes",
    "breast_cancer",
    "heart_disease",
]


# ============================================================
# 1. SELECT THE DECISION THRESHOLD USING INNER OOF SCORES
# ============================================================

def choose_threshold(y_true, oof_scores):
    y_array = np.asarray(y_true, dtype=int)
    scores = np.asarray(oof_scores, dtype=float)

    unique_scores = np.unique(scores)

    if unique_scores.size == 1:
        candidates = np.array(
            [
                unique_scores[0] - 1.0,
                unique_scores[0],
                unique_scores[0] + 1.0,
            ]
        )
    else:
        midpoints = (
            unique_scores[:-1] + unique_scores[1:]
        ) / 2.0

        span = max(
            float(unique_scores[-1] - unique_scores[0]),
            1.0,
        )

        candidates = np.concatenate(
            (
                [unique_scores[0] - span],
                midpoints,
                [unique_scores[-1] + span],
            )
        )

    scored_thresholds = []

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)

        score = balanced_accuracy_score(
            y_array,
            predictions,
        )

        scored_thresholds.append(
            (float(score), float(threshold))
        )

    best_score = max(
        item[0] for item in scored_thresholds
    )

    tied = [
        item
        for item in scored_thresholds
        if np.isclose(item[0], best_score)
    ]

    # Tie-break: prefer the threshold closest to zero.
    best_threshold = min(
        tied,
        key=lambda item: (abs(item[1]), item[1]),
    )[1]

    return best_threshold, best_score


# ============================================================
# 2. CALIBRATE AND EVALUATE ONE SELECTED CANDIDATE
# ============================================================

def evaluate_calibrated_candidate(
    candidate,
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    method_name,
    outer_fold,
    numeric_imputation,
    dimensionality_reduction,
):
    model_args = {
        "C": candidate["C"],
        "gamma": candidate["gamma"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_imputation": numeric_imputation,
        "dimensionality_reduction": dimensionality_reduction,
    }

    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    inner_splits = list(
        inner_cv.split(X_train, y_train)
    )

    # Generate out-of-fold decision scores on outer-training data.
    calibration_model = create_model(**model_args)

    oof_scores = cross_val_predict(
        calibration_model,
        X_train,
        y_train,
        cv=inner_splits,
        method="decision_function",
        n_jobs=1,
    )

    threshold, inner_oof_balanced_accuracy = choose_threshold(
        y_train,
        oof_scores,
    )

    # Fit on all outer-training data.
    # The outer test fold does not participate in calibration.
    final_model = create_model(**model_args)
    final_model.fit(X_train, y_train)

    test_scores = final_model.decision_function(X_test)
    predictions = (test_scores >= threshold).astype(int)

    y_test_array = np.asarray(y_test, dtype=int)

    outer_auc = roc_auc_score(
        y_test_array,
        test_scores,
    )

    balanced_acc = balanced_accuracy_score(
        y_test_array,
        predictions,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = f1_score(
        y_test_array,
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
        "threshold": threshold,
        "inner_oof_balanced_accuracy_at_threshold":
            inner_oof_balanced_accuracy,
        "C": candidate["C"],
        "gamma": candidate["gamma"],
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
        "predicted_negative_count": int(
            (predictions == 0).sum()
        ),
        "predicted_positive_count": int(
            (predictions == 1).sum()
        ),
        "decision_score_min": float(np.min(test_scores)),
        "decision_score_max": float(np.max(test_scores)),
    }


# ============================================================
# 3. RUN ONE DATASET + ONE CONFIGURATION
# ============================================================

def run_calibrated_experiment(dataset_name, config):
    config_name = config["name"]
    numeric_imputation = config["numeric_imputation"]
    dimensionality_reduction = config["dimensionality_reduction"]

    print("\n" + "#" * 72)
    print(f"CALIBRATED DATASET: {dataset_name}")
    print(f"CONFIGURATION: {config_name}")
    print("#" * 72)

    X, y, categorical_features = get_dataset(dataset_name)

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for outer_fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        print(
            f"\n{dataset_name} | {config_name} | "
            f"outer fold {outer_fold}/{OUTER_FOLDS}"
        )

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

        selected = select_candidates(candidates)

        for method_name, candidate in selected.items():
            metrics = evaluate_calibrated_candidate(
                candidate=candidate,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                method_name=method_name,
                outer_fold=outer_fold,
                numeric_imputation=numeric_imputation,
                dimensionality_reduction=dimensionality_reduction,
            )

            row = {
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
                "threshold_calibration":
                    "inner_oof_balanced_accuracy",
                **metrics,
            }

            rows.append(row)

            print(
                f"{method_name}: "
                f"threshold={metrics['threshold']:.5f}, "
                f"AUC={metrics['outer_roc_auc']:.4f}, "
                f"balanced accuracy="
                f"{metrics['balanced_accuracy']:.4f}, "
                f"sensitivity={metrics['sensitivity']:.4f}, "
                f"specificity={metrics['specificity']:.4f}, "
                f"F1={metrics['f1_score']:.4f}"
            )

    results_df = pd.DataFrame(rows)

    output_path = RESULTS_DIR / (
        f"nested_cv_{dataset_name}_{config_name}_"
        "threshold_calibrated_outer_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    metric_columns = [
        "outer_roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "overtuning_gap",
    ]

    print("\nMean calibrated outer-fold metrics:")
    print(
        results_df.groupby("method")[metric_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print(f"\nResults saved to: {output_path}")

    return results_df


# ============================================================
# 4. MAIN: RUN CALIBRATION ON ALL DATASETS
# ============================================================

def main():
    all_results = []
    completed_datasets = []
    failed_datasets = []

    print("=" * 72)
    print("MULTI-DATASET THRESHOLD-CALIBRATION EXPERIMENT")
    print("=" * 72)

    for dataset_name in DATASETS:
        print("\n" + "=" * 72)
        print(f"STARTING CALIBRATION FOR: {dataset_name}")
        print("=" * 72)

        try:
            # Validate the dataset before starting the search.
            X, y, _ = get_dataset(dataset_name)

            if y.nunique() != 2:
                raise ValueError(
                    f"{dataset_name} must have exactly two target classes."
                )

            dataset_results = []

            for config in PREPROCESSING_CONFIGS:
                result_df = run_calibrated_experiment(
                    dataset_name,
                    config,
                )

                dataset_results.append(result_df)

            combined = pd.concat(
                dataset_results,
                ignore_index=True,
            )

            combined_path = RESULTS_DIR / (
                f"nested_cv_{dataset_name}_"
                "threshold_calibrated_comparison.csv"
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
                f"nested_cv_{dataset_name}_"
                "threshold_calibrated_summary.csv"
            )

            summary.to_csv(summary_path)

            all_results.append(combined)
            completed_datasets.append(dataset_name)

            print(f"\nCompleted: {dataset_name}")
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
            print("Continuing to the next dataset.")

    if all_results:
        all_datasets_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined_path = RESULTS_DIR / (
            "nested_cv_all_datasets_threshold_calibrated_comparison.csv"
        )

        all_datasets_df.to_csv(
            combined_path,
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

        summary_path = RESULTS_DIR / (
            "nested_cv_all_datasets_threshold_calibrated_summary.csv"
        )

        cross_dataset_summary.to_csv(summary_path)

        print("\n" + "=" * 72)
        print("CROSS-DATASET CALIBRATED SUMMARY")
        print("=" * 72)
        print(cross_dataset_summary.to_string())
        print(f"\nCombined results: {combined_path}")
        print(f"Combined summary: {summary_path}")

    print("\nCompleted datasets:", completed_datasets)
    print("Failed datasets:", failed_datasets)


if __name__ == "__main__":
    main()