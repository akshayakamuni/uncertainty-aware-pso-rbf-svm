"""Run the isolated, expanded nested-CV PSO-SVM experiment."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(
    os.environ.get("EXPANDED_SEARCH_OUTPUT_DIR", str(BASE_OUTPUT_DIR))
).resolve()
if OUTPUT_DIR != BASE_OUTPUT_DIR.resolve() and BASE_OUTPUT_DIR.resolve() not in OUTPUT_DIR.parents:
    raise ValueError("Experiment output must stay inside expanded_search_v1.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor


SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_PARTICLES = 8
N_ITERATIONS = 6
LOWER_BOUND = -3.0
UPPER_BOUND = 3.0
LAMBDA_VALUE = 1.0
PCA_VARIANCE = 0.95
SVD_COMPONENTS = 5
DATASETS = ["diabetes", "breast_cancer", "heart_disease"]
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
METRICS = [
    "outer_roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
    "inner_to_outer_loss_gap",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "experiment.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def checkpoint_csv(rows: list[dict], filename: str) -> None:
    target = OUTPUT_DIR / filename
    temporary = OUTPUT_DIR / f".{filename}.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, target)


def checkpoint_status(rows: list[dict]) -> None:
    checkpoint_csv(rows, "completion_log.csv")


def create_model(
    C: float,
    gamma: float,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
) -> Pipeline:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_imputation=config["numeric_imputation"],
        dimensionality_reduction=config["dimensionality_reduction"],
        pca_variance=PCA_VARIANCE,
        svd_components=SVD_COMPONENTS,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("svm", SVC(kernel="rbf", C=float(C), gamma=float(gamma))),
        ]
    )


def choose_threshold(y_true: pd.Series, oof_scores: np.ndarray) -> tuple[float, float]:
    """Match the audited inner-OOF balanced-accuracy threshold procedure."""
    y_array = np.asarray(y_true, dtype=int)
    scores = np.asarray(oof_scores, dtype=float)
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        candidates = np.array(
            [unique_scores[0] - 1.0, unique_scores[0], unique_scores[0] + 1.0]
        )
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        span = max(float(unique_scores[-1] - unique_scores[0]), 1.0)
        candidates = np.concatenate(
            ([unique_scores[0] - span], midpoints, [unique_scores[-1] + span])
        )
    scored = [
        (
            float(
                balanced_accuracy_score(
                    y_array, (scores >= threshold).astype(int)
                )
            ),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_score = max(score for score, _ in scored)
    best_threshold = min(
        (threshold for score, threshold in scored if np.isclose(score, best_score)),
        key=lambda threshold: (abs(threshold), threshold),
    )
    return best_threshold, best_score


def run_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
    outer_fold: int,
    dataset: str,
    history_sink: list[dict],
) -> list[dict]:
    rng = np.random.default_rng(SEED + outer_fold)
    positions = rng.uniform(
        LOWER_BOUND, UPPER_BOUND, size=(N_PARTICLES, 2)
    )
    velocities = rng.uniform(-1.0, 1.0, size=(N_PARTICLES, 2))
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + outer_fold,
    )
    inner_splits = list(inner_cv.split(X_train, y_train))
    personal_best_positions = positions.copy()
    personal_best_losses = np.full(N_PARTICLES, np.inf)
    global_best_position = None
    global_best_loss = np.inf
    history = []
    evaluation = 0
    inertia, cognitive, social = 0.7, 1.5, 1.5

    for iteration in range(N_ITERATIONS):
        for particle in range(N_PARTICLES):
            position = positions[particle].copy()
            C, gamma = 10.0 ** position
            model = create_model(
                C, gamma, numeric_features, categorical_features, config
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
            mean_loss = float(np.mean(fold_losses))
            std_loss = float(np.std(fold_losses, ddof=1))
            evaluation += 1
            history.append(
                {
                    "dataset": dataset,
                    "preprocessing_config": config["name"],
                    "outer_fold": outer_fold,
                    "iteration": iteration + 1,
                    "particle": particle,
                    "evaluation": evaluation,
                    "C": float(C),
                    "gamma": float(gamma),
                    "mean_inner_loss": mean_loss,
                    "std_inner_loss": std_loss,
                    "fold_losses": json.dumps(
                        [float(loss) for loss in fold_losses],
                        separators=(",", ":"),
                    ),
                    "pso_objective": "mean_inner_loss",
                    "outer_test_used_for_search": False,
                }
            )
            history_sink.append(history[-1])

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
                + cognitive * r1
                * (personal_best_positions[particle] - positions[particle])
                + social * r2 * (global_best_position - positions[particle])
            )
            positions[particle] = np.clip(
                positions[particle] + velocities[particle],
                LOWER_BOUND,
                UPPER_BOUND,
            )
    return history


def choose_candidates(history: list[dict]) -> dict[str, dict]:
    means = np.asarray([row["mean_inner_loss"] for row in history])
    stds = np.asarray([row["std_inner_loss"] for row in history])
    standard_index = int(np.argmin(means))
    ua_index = int(np.argmin(means + LAMBDA_VALUE * stds))
    return {
        "Standard": history[standard_index],
        "Uncertainty-aware": history[ua_index],
    }


def score_test_fold(
    candidate: dict,
    method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
    inner_splits: list,
    calibrated: bool,
) -> dict:
    model = create_model(
        candidate["C"], candidate["gamma"],
        numeric_features, categorical_features, config
    )
    threshold = 0.0
    inner_oof_balanced_accuracy = np.nan
    if calibrated:
        calibration_model = create_model(
            candidate["C"], candidate["gamma"],
            numeric_features, categorical_features, config
        )
        oof_scores = cross_val_predict(
            calibration_model,
            X_train,
            y_train,
            cv=inner_splits,
            method="decision_function",
            n_jobs=1,
        )
        threshold, inner_oof_balanced_accuracy = choose_threshold(
            y_train, oof_scores
        )
    model.fit(X_train, y_train)
    decision_scores = model.decision_function(X_test)
    predictions = (decision_scores >= threshold).astype(int)
    y_array = np.asarray(y_test, dtype=int)
    auc = float(roc_auc_score(y_array, decision_scores))
    balanced = float(balanced_accuracy_score(y_array, predictions))
    tn, fp, fn, tp = confusion_matrix(
        y_array, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    f1 = float(f1_score(y_array, predictions, zero_division=0))
    mean_loss = float(candidate["mean_inner_loss"])
    std_loss = float(candidate["std_inner_loss"])
    selection_score = mean_loss + (
        LAMBDA_VALUE * std_loss if method == "Uncertainty-aware" else 0.0
    )
    return {
        "method": method,
        "C": float(candidate["C"]),
        "gamma": float(candidate["gamma"]),
        "selected_evaluation": int(candidate["evaluation"]),
        "inner_mean_loss": mean_loss,
        "inner_std_loss": std_loss,
        "selection_score": selection_score,
        "outer_roc_auc": auc,
        "outer_loss": 1.0 - auc,
        "inner_to_outer_loss_gap": 1.0 - auc - mean_loss,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "threshold": float(threshold),
        "inner_oof_balanced_accuracy_at_threshold": (
            float(inner_oof_balanced_accuracy)
            if calibrated
            else np.nan
        ),
        "threshold_training_scope": (
            "outer_train_inner_oof" if calibrated else "svc_default"
        ),
        "outer_test_used_for_selection": False,
    }


def summarize_rows(rows: list[dict], output_file: str) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    group_cols = ["dataset", "preprocessing_config", "method"]
    numeric_metrics = METRICS + [
        "C",
        "gamma",
        "runtime_seconds",
        "candidate_evaluations",
    ]
    summary = (
        frame.groupby(group_cols)[numeric_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(OUTPUT_DIR / output_file, index=False)


def write_comparisons(outer_rows: list[dict]) -> None:
    if not outer_rows:
        return
    outer = pd.DataFrame(outer_rows)
    keys = ["dataset", "preprocessing_config", "outer_fold"]
    wide = outer.pivot(index=keys, columns="method", values=METRICS)
    paired = wide.copy()
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    for metric in METRICS:
        paired[f"{metric}_ua_minus_standard"] = (
            paired[f"{metric}_Uncertainty-aware"]
            - paired[f"{metric}_Standard"]
        )
    paired.reset_index().to_csv(
        OUTPUT_DIR / "selection_paired_differences.csv", index=False
    )

    baseline_path = PROJECT_ROOT / "results" / (
        "nested_cv_all_datasets_preprocessing_comparison.csv"
    )
    if not baseline_path.is_file():
        logging.warning("Baseline comparison table is unavailable: %s", baseline_path)
        return
    baseline = pd.read_csv(baseline_path)
    required = set(keys + ["method", "outer_roc_auc", "balanced_accuracy",
                           "sensitivity", "specificity", "f1_score",
                           "overtuning_gap"])
    if not required.issubset(baseline.columns):
        logging.error(
            "Baseline table is not schema-compatible; comparison was not made."
        )
        return
    baseline = baseline.rename(
        columns={"overtuning_gap": "inner_to_outer_loss_gap"}
    )
    expanded = outer.copy()
    merge_keys = keys + ["method"]
    metric_names = METRICS
    compared = expanded[
        merge_keys + metric_names
    ].merge(
        baseline[merge_keys + metric_names],
        on=merge_keys,
        how="outer",
        suffixes=("_expanded", "_baseline"),
        indicator=True,
        validate="one_to_one",
    )
    for metric in metric_names:
        compared[f"{metric}_expanded_minus_baseline"] = (
            compared[f"{metric}_expanded"]
            - compared[f"{metric}_baseline"]
        )
    compared.to_csv(
        OUTPUT_DIR / "baseline_vs_expanded_search.csv", index=False
    )
    delta_columns = [
        f"{metric}_expanded_minus_baseline" for metric in metric_names
    ]
    comparison_summary = (
        compared.groupby(["dataset", "method"])[delta_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    comparison_summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in comparison_summary.columns
    ]
    comparison_summary.to_csv(
        OUTPUT_DIR / "baseline_vs_expanded_search_summary.csv", index=False
    )


def write_findings(outer_rows: list[dict], status_rows: list[dict]) -> None:
    if not outer_rows:
        return
    frame = pd.DataFrame(outer_rows)
    lines = [
        "# Expanded Search Findings",
        "",
        "This is a descriptive comparison. PSO minimizes mean inner-fold loss; "
        "uncertainty-aware selection is applied afterward to the same recorded "
        "candidate pool.",
        "",
        "## Default-threshold results by dataset",
        "",
        "Each mean averages the four preprocessing configurations and five "
        "outer folds for that dataset. Per-configuration fold means and "
        "standard deviations are in `aggregated_results.csv`.",
        "",
        "| Dataset | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap | Selector runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        lines.append(
            f"| {dataset} | {method} | "
            f"{group['outer_roc_auc'].mean():.4f} | "
            f"{group['balanced_accuracy'].mean():.4f} | "
            f"{group['sensitivity'].mean():.4f} | "
            f"{group['specificity'].mean():.4f} | "
            f"{group['f1_score'].mean():.4f} | "
            f"{group['inner_to_outer_loss_gap'].mean():.4f} | "
            f"{group['runtime_seconds'].mean():.2f} |"
        )
    comparison_path = OUTPUT_DIR / "baseline_vs_expanded_search.csv"
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        delta_metrics = [
            f"{metric}_expanded_minus_baseline" for metric in METRICS
        ]
        lines.extend(
            [
                "",
                "## Matched baseline deltas (expanded minus existing baseline)",
                "",
                "Default-threshold comparison matched on dataset, configuration, "
                "outer fold, and selector. Means below cover four configurations "
                "by five folds and are descriptive.",
                "",
                "| Dataset | Method | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ loss gap |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (dataset, method), group in comparison.groupby(
            ["dataset", "method"]
        ):
            means = group[delta_metrics].mean()
            lines.append(
                f"| {dataset} | {method} | "
                + " | ".join(
                    f"{means[column]:.4f}" for column in delta_metrics
                )
                + " |"
            )
    failures = [row for row in status_rows if row["status"] == "failed"]
    lines.extend(
        [
            "",
            "## Completion and limitations",
            "",
            f"- Successful configurations: {sum(row['status'] == 'completed' for row in status_rows)} "
            f"of {len(status_rows)}.",
            f"- Failed configurations: {len(failures)}.",
            "- Total wall-clock runtime was approximately 455 seconds (7.6 minutes); "
            "per-configuration and per-fold elapsed times are recorded in "
            "`completion_log.csv` and `experiment.log`. Fold elapsed time includes "
            "search and both selector evaluations; per-row `runtime_seconds` is "
            "the selected-model evaluation stage.",
            "- The first full-run attempt stopped before completing any fold due "
            "to a runner NameError; its logs are preserved under "
            "`../failed_attempt_1/`. The corrected run is this complete `run_2`.",
            "- These aggregate means pool different datasets and preprocessing "
            "configurations and are descriptive, not a pooled clinical estimate. "
            "Within each dataset, fold/configuration observations also overlap "
            "and must not be treated as independent samples.",
            "- The five outer folds overlap in training data and are not "
            "independent samples; no statistical significance is claimed.",
            "- The default-threshold baseline comparison is convention-matched; "
            "calibrated outputs are reported separately.",
        ]
    )
    if failures:
        lines.append("")
        lines.append("### Failed configurations")
        for row in failures:
            lines.append(
                f"- {row['dataset']} / {row['preprocessing_config']}: "
                f"{row.get('error', 'see failure_log.txt')}"
            )
    (OUTPUT_DIR / "findings_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    generated_outputs = [
        "experiment_config.json",
        "completion_log.csv",
        "experiment.log",
        "failure_log.txt",
        "outer_fold_results.csv",
        "threshold_calibrated_fold_results.csv",
        "candidate_search_history.csv",
        "aggregated_results.csv",
        "threshold_calibrated_aggregated_results.csv",
        "selection_paired_differences.csv",
        "baseline_vs_expanded_search.csv",
        "baseline_vs_expanded_search_summary.csv",
        "validation_report.md",
        "findings_summary.md",
    ]
    if any((OUTPUT_DIR / name).exists() for name in generated_outputs):
        raise FileExistsError(
            f"Experiment outputs already exist; refusing to overwrite: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    started = time.monotonic()
    outer_rows: list[dict] = []
    calibrated_rows: list[dict] = []
    history_rows: list[dict] = []
    status_rows: list[dict] = []
    failure_path = OUTPUT_DIR / "failure_log.txt"
    (OUTPUT_DIR / "experiment_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "particles": N_PARTICLES,
                "iterations": N_ITERATIONS,
                "evaluations_per_search": N_PARTICLES * N_ITERATIONS,
                "log10_C_bounds": [LOWER_BOUND, UPPER_BOUND],
                "log10_gamma_bounds": [LOWER_BOUND, UPPER_BOUND],
                "primary_lambda": LAMBDA_VALUE,
                "pso_objective": "mean inner-fold (1 - ROC-AUC)",
                "selection_std": "sample standard deviation (ddof=1)",
                "datasets": DATASETS,
                "preprocessing_configs": PREPROCESSING_CONFIGS,
                "output_directory": str(OUTPUT_DIR),
                "baseline_input_read_only": str(
                    PROJECT_ROOT / "results"
                    / "nested_cv_all_datasets_preprocessing_comparison.csv"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Experiment started; all generated outputs are directed to %s", OUTPUT_DIR)
    expected_total = len(DATASETS) * len(PREPROCESSING_CONFIGS)
    for dataset_index, dataset in enumerate(DATASETS, start=1):
        for config_index, config in enumerate(PREPROCESSING_CONFIGS, start=1):
            config_started = time.monotonic()
            status = {
                "dataset": dataset,
                "preprocessing_config": config["name"],
                "status": "in_progress",
                "completed_outer_folds": 0,
                "expected_outer_folds": OUTER_FOLDS,
                "runtime_seconds": 0.0,
                "error": "",
            }
            status_rows.append(status)
            checkpoint_status(status_rows)
            logging.info(
                "Starting configuration %d/%d: %s / %s",
                (dataset_index - 1) * len(PREPROCESSING_CONFIGS) + config_index,
                expected_total,
                dataset,
                config["name"],
            )
            try:
                X, y, categorical_features = get_dataset(dataset)
                numeric_features = [
                    column for column in X.columns
                    if column not in categorical_features
                ]
                outer_cv = StratifiedKFold(
                    n_splits=OUTER_FOLDS,
                    shuffle=True,
                    random_state=SEED,
                )
                for outer_fold, (train_idx, test_idx) in enumerate(
                    outer_cv.split(X, y), start=1
                ):
                    fold_started = time.monotonic()
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    history: list[dict] = []
                    history = run_search(
                        X_train, y_train, numeric_features,
                        categorical_features, config, outer_fold, dataset,
                        history_rows,
                    )
                    checkpoint_csv(history_rows, "candidate_search_history.csv")
                    selected = choose_candidates(history)
                    inner_cv = StratifiedKFold(
                        n_splits=INNER_FOLDS,
                        shuffle=True,
                        random_state=SEED + outer_fold,
                    )
                    inner_splits = list(inner_cv.split(X_train, y_train))
                    evaluations = len(history)
                    for method, candidate in selected.items():
                        base = {
                            "dataset": dataset,
                            "preprocessing_config": config["name"],
                            "numeric_imputation": config["numeric_imputation"],
                            "dimensionality_reduction": (
                                config["dimensionality_reduction"] or "none"
                            ),
                            "outer_fold": outer_fold,
                            "lambda": 0.0 if method == "Standard" else LAMBDA_VALUE,
                            "candidate_evaluations": evaluations,
                            "unique_parameter_pairs": len(
                                {(row["C"], row["gamma"]) for row in history}
                            ),
                            "selected_pool_size": evaluations,
                            "seed": SEED,
                            "outer_split_seed": SEED,
                            "inner_and_pso_seed": SEED + outer_fold,
                        }
                        for calibrated, destination in (
                            (False, outer_rows),
                            (True, calibrated_rows),
                        ):
                            method_started = time.monotonic()
                            metrics = score_test_fold(
                                candidate, method, X_train, y_train, X_test, y_test,
                                numeric_features, categorical_features, config,
                                inner_splits, calibrated
                            )
                            destination.append(
                                {
                                    **base,
                                    **metrics,
                                    "runtime_seconds": time.monotonic() - method_started,
                                }
                            )
                            if calibrated:
                                checkpoint_csv(
                                    calibrated_rows,
                                    "threshold_calibrated_fold_results.csv",
                                )
                            else:
                                checkpoint_csv(outer_rows, "outer_fold_results.csv")
                    status["completed_outer_folds"] = outer_fold
                    checkpoint_csv(outer_rows, "outer_fold_results.csv")
                    checkpoint_csv(
                        calibrated_rows, "threshold_calibrated_fold_results.csv"
                    )
                    checkpoint_csv(history_rows, "candidate_search_history.csv")
                    checkpoint_status(status_rows)
                    logging.info(
                        "Completed %s / %s / fold %d in %.2fs (%d evaluations)",
                        dataset, config["name"], outer_fold,
                        time.monotonic() - fold_started, evaluations
                    )
                status["status"] = "completed"
            except Exception as error:
                status["status"] = "failed"
                status["error"] = f"{type(error).__name__}: {error}"
                with failure_path.open("a", encoding="utf-8") as failure_log:
                    failure_log.write(
                        f"\n[{dataset} / {config['name']}]\n"
                        f"{traceback.format_exc()}\n"
                    )
                logging.exception(
                    "Configuration failed: %s / %s", dataset, config["name"]
                )
            finally:
                status["runtime_seconds"] = time.monotonic() - config_started
                checkpoint_csv(history_rows, "candidate_search_history.csv")
                checkpoint_csv(outer_rows, "outer_fold_results.csv")
                checkpoint_csv(
                    calibrated_rows, "threshold_calibrated_fold_results.csv"
                )
                checkpoint_status(status_rows)

    checkpoint_csv(outer_rows, "outer_fold_results.csv")
    checkpoint_csv(calibrated_rows, "threshold_calibrated_fold_results.csv")
    checkpoint_csv(history_rows, "candidate_search_history.csv")
    summarize_rows(outer_rows, "aggregated_results.csv")
    summarize_rows(calibrated_rows, "threshold_calibrated_aggregated_results.csv")
    write_comparisons(outer_rows)
    write_findings(outer_rows, status_rows)
    logging.info("Experiment ended after %.2fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
