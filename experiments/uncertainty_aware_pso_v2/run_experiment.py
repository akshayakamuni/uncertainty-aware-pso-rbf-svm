"""Independent four-method uncertainty-aware PSO-SVM experiment."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
OUTPUT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataset
from src.preprocessing import build_preprocessor


SEED = 42
SVD_RANDOM_STATE = SEED
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_PARTICLES = 8
N_ITERATIONS = 6
EVALUATIONS_PER_SEARCH = N_PARTICLES * N_ITERATIONS
LOWER_BOUND = -3.0
UPPER_BOUND = 3.0
PRIMARY_LAMBDA = 1.0
LAMBDA_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0]
PCA_VARIANCE = 0.95
SVD_COMPONENTS = 5
DATASETS = ["diabetes", "breast_cancer", "heart_disease"]
PREPROCESSING_CONFIGS = [
    {"name": "baseline_median", "numeric_imputation": "median", "dimensionality_reduction": None},
    {"name": "mean_imputation", "numeric_imputation": "mean", "dimensionality_reduction": None},
    {"name": "pca_median", "numeric_imputation": "median", "dimensionality_reduction": "pca"},
    {"name": "svd_median", "numeric_imputation": "median", "dimensionality_reduction": "svd"},
]
METRICS = [
    "outer_roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
    "inner_to_outer_loss_gap",
]
RESULT_FILES = [
    "experiment_config.json",
    "outer_fold_results.csv",
    "candidate_search_history.csv",
    "threshold_calibrated_fold_results.csv",
    "aggregated_results.csv",
    "paired_method_comparisons.csv",
    "lambda_sensitivity.csv",
    "runtime_summary.csv",
    "completion_log.csv",
    "experiment.log",
    "failure_log.txt",
    "validation_report.md",
    "findings_summary.md",
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


def checkpoint(rows: list[dict], filename: str) -> None:
    target = OUTPUT_DIR / filename
    temp = OUTPUT_DIR / f".{filename}.tmp"
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, target)


def fitness_from_losses(fold_losses: Any, lambda_value: float) -> dict[str, float]:
    losses = np.asarray(fold_losses, dtype=float)
    if losses.ndim != 1 or losses.size < 2:
        raise ValueError("Each candidate requires at least two inner-fold losses.")
    if not np.isfinite(losses).all():
        raise ValueError("Inner-fold losses must all be finite.")
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and non-negative.")
    mean_loss = float(np.mean(losses))
    sample_sd = float(np.std(losses, ddof=1))
    return {
        "mean_loss": mean_loss,
        "sample_sd": sample_sd,
        "fitness": mean_loss + lambda_value * sample_sd,
    }


def update_best(
    candidate_fitness: float,
    candidate_evaluation: int,
    best_fitness: float,
    best_evaluation: int,
) -> tuple[float, int]:
    if candidate_fitness < best_fitness:
        return float(candidate_fitness), int(candidate_evaluation)
    return float(best_fitness), int(best_evaluation)


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
    if config["dimensionality_reduction"] == "svd":
        preprocessor.set_params(reduction__random_state=SVD_RANDOM_STATE)
    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("svm", SVC(kernel="rbf", C=float(C), gamma=float(gamma))),
        ]
    )


def evaluate_position(
    position: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
    inner_splits: list,
) -> tuple[float, float, list[float], float]:
    started = time.monotonic()
    C, gamma = np.power(10.0, position)
    model = create_model(
        float(C), float(gamma), numeric_features, categorical_features, config
    )
    aucs = cross_val_score(
        model,
        X_train,
        y_train,
        cv=inner_splits,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    )
    losses = 1.0 - aucs
    stats = fitness_from_losses(losses, 0.0)
    return float(C), float(gamma), [float(loss) for loss in losses], time.monotonic() - started


def _record_candidate(
    rows: list[dict],
    *,
    dataset: str,
    config: dict,
    outer_fold: int,
    search_label: str,
    method: str,
    iteration: int,
    particle: int,
    evaluation: int,
    C: float,
    gamma: float,
    losses: list[float],
    lambda_value: float,
    evaluation_seconds: float,
    pso_fitness_kind: str,
    personal_best_eval: int | None = None,
    personal_best_fitness: float | None = None,
    global_best_eval: int | None = None,
    global_best_fitness: float | None = None,
) -> dict:
    stats = fitness_from_losses(losses, lambda_value)
    row = {
        "dataset": dataset,
        "preprocessing_config": config["name"],
        "outer_fold": outer_fold,
        "search_label": search_label,
        "method_family": method,
        "iteration": iteration,
        "particle": particle,
        "evaluation": evaluation,
        "C": C,
        "gamma": gamma,
        "log10_C": float(np.log10(C)),
        "log10_gamma": float(np.log10(gamma)),
        "fold_losses": json.dumps(losses, separators=(",", ":")),
        "mean_inner_loss": stats["mean_loss"],
        "std_inner_loss": stats["sample_sd"],
        "lambda": lambda_value,
        "fitness": stats["fitness"],
        "pso_fitness_kind": pso_fitness_kind,
        "personal_best_evaluation": personal_best_eval,
        "personal_best_fitness": personal_best_fitness,
        "global_best_evaluation": global_best_eval,
        "global_best_fitness": global_best_fitness,
        "evaluation_runtime_seconds": evaluation_seconds,
        "outer_test_used_for_search": False,
        "selected_methods": "",
    }
    rows.append(row)
    return row


def run_pso_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
    dataset: str,
    outer_fold: int,
    lambda_value: float,
    search_label: str,
    candidate_rows: list[dict],
    inner_splits: list,
) -> tuple[list[dict], float]:
    started = time.monotonic()
    rng = np.random.default_rng(SEED + outer_fold)
    positions = rng.uniform(
        LOWER_BOUND, UPPER_BOUND, size=(N_PARTICLES, 2)
    )
    velocities = rng.uniform(-1.0, 1.0, size=(N_PARTICLES, 2))
    initial_positions = positions.copy()
    personal_best_positions = positions.copy()
    personal_best_fitness = np.full(N_PARTICLES, np.inf)
    personal_best_evaluation = np.zeros(N_PARTICLES, dtype=int)
    global_best_position: np.ndarray | None = None
    global_best_fitness = np.inf
    global_best_evaluation = 0
    history: list[dict] = []
    evaluation = 0
    inertia, cognitive, social = 0.7, 1.5, 1.5

    for iteration in range(1, N_ITERATIONS + 1):
        for particle in range(N_PARTICLES):
            position = positions[particle].copy()
            C, gamma, losses, evaluation_seconds = evaluate_position(
                position, X_train, y_train, numeric_features,
                categorical_features, config, inner_splits
            )
            evaluation += 1
            stats = fitness_from_losses(losses, lambda_value)
            personal_best_fitness[particle], personal_best_evaluation[particle] = (
                update_best(
                    stats["fitness"],
                    evaluation,
                    personal_best_fitness[particle],
                    personal_best_evaluation[particle],
                )
            )
            if personal_best_evaluation[particle] == evaluation:
                personal_best_positions[particle] = position.copy()
            global_best_fitness, global_best_evaluation = update_best(
                stats["fitness"],
                evaluation,
                global_best_fitness,
                global_best_evaluation,
            )
            if global_best_evaluation == evaluation:
                global_best_position = position.copy()
            row = _record_candidate(
                candidate_rows,
                dataset=dataset,
                config=config,
                outer_fold=outer_fold,
                search_label=search_label,
                method="PSO",
                iteration=iteration,
                particle=particle,
                evaluation=evaluation,
                C=C,
                gamma=gamma,
                losses=losses,
                lambda_value=lambda_value,
                evaluation_seconds=evaluation_seconds,
                pso_fitness_kind=(
                    "mean_plus_lambda_sample_sd"
                    if lambda_value else "mean_loss"
                ),
                personal_best_eval=int(personal_best_evaluation[particle]),
                personal_best_fitness=float(personal_best_fitness[particle]),
                global_best_eval=int(global_best_evaluation),
                global_best_fitness=float(global_best_fitness),
            )
            row["initial_log10_C"] = float(initial_positions[particle, 0])
            row["initial_log10_gamma"] = float(initial_positions[particle, 1])
            history.append(row)
        if global_best_position is None:
            raise RuntimeError("PSO did not establish a finite global best.")
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

    if len(history) != EVALUATIONS_PER_SEARCH:
        raise RuntimeError(
            f"{search_label} produced {len(history)} evaluations; "
            f"expected {EVALUATIONS_PER_SEARCH}."
        )
    return history, time.monotonic() - started


def run_random_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict,
    dataset: str,
    outer_fold: int,
    candidate_rows: list[dict],
    inner_splits: list,
) -> tuple[list[dict], float]:
    started = time.monotonic()
    seed = SEED + outer_fold + 100_000
    rng = np.random.default_rng(seed)
    history = []
    for evaluation in range(1, EVALUATIONS_PER_SEARCH + 1):
        position = rng.uniform(LOWER_BOUND, UPPER_BOUND, size=2)
        C, gamma, losses, evaluation_seconds = evaluate_position(
            position, X_train, y_train, numeric_features,
            categorical_features, config, inner_splits
        )
        history.append(
            _record_candidate(
                candidate_rows,
                dataset=dataset,
                config=config,
                outer_fold=outer_fold,
                search_label="M4_random_search",
                method="Random",
                iteration=1,
                particle=evaluation - 1,
                evaluation=evaluation,
                C=C,
                gamma=gamma,
                losses=losses,
                lambda_value=0.0,
                evaluation_seconds=evaluation_seconds,
                pso_fitness_kind="mean_loss",
            )
        )
    return history, time.monotonic() - started


def select_candidate(pool: list[dict], criterion_lambda: float) -> dict:
    if not pool:
        raise ValueError("Cannot select from an empty candidate pool.")
    return min(
        pool,
        key=lambda row: (
            row["mean_inner_loss"] + criterion_lambda * row["std_inner_loss"],
            row["evaluation"],
        ),
    )


def choose_threshold(y_true: pd.Series, oof_scores: np.ndarray) -> tuple[float, float]:
    y_array = np.asarray(y_true, dtype=int)
    scores = np.asarray(oof_scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Inner OOF decision scores must be finite.")
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        thresholds = np.array(
            [unique_scores[0] - 1.0, unique_scores[0], unique_scores[0] + 1.0]
        )
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        span = max(float(unique_scores[-1] - unique_scores[0]), 1.0)
        thresholds = np.concatenate(
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
        for threshold in thresholds
    ]
    best_score = max(score for score, _ in scored)
    best_threshold = min(
        (threshold for score, threshold in scored if np.isclose(score, best_score)),
        key=lambda threshold: (abs(threshold), threshold),
    )
    return best_threshold, best_score


def score_candidate(
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
    calibrated: bool = False,
    selection_lambda: float | None = None,
) -> tuple[dict, float]:
    started = time.monotonic()
    threshold = 0.0
    inner_oof_ba = np.nan
    if calibrated:
        oof_model = create_model(
            candidate["C"], candidate["gamma"],
            numeric_features, categorical_features, config
        )
        oof_scores = cross_val_predict(
            oof_model,
            X_train,
            y_train,
            cv=inner_splits,
            method="decision_function",
            n_jobs=1,
        )
        threshold, inner_oof_ba = choose_threshold(y_train, oof_scores)

    final_model = create_model(
        candidate["C"], candidate["gamma"],
        numeric_features, categorical_features, config
    )
    final_model.fit(X_train, y_train)
    test_scores = final_model.decision_function(X_test)
    predictions = (
        (test_scores >= threshold).astype(int)
        if calibrated
        else final_model.predict(X_test)
    )
    y_array = np.asarray(y_test, dtype=int)
    auc = float(roc_auc_score(y_array, test_scores))
    balanced = float(balanced_accuracy_score(y_array, predictions))
    tn, fp, fn, tp = confusion_matrix(
        y_array, predictions, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    f1 = float(f1_score(y_array, predictions, zero_division=0))
    mean_loss = float(candidate["mean_inner_loss"])
    fitness_lambda = (
        float(candidate["lambda"])
        if selection_lambda is None
        else float(selection_lambda)
    )
    selection_score = mean_loss + fitness_lambda * float(candidate["std_inner_loss"])
    return (
        {
            "method": method,
            "C": float(candidate["C"]),
            "gamma": float(candidate["gamma"]),
            "selected_evaluation": int(candidate["evaluation"]),
            "search_label": candidate["search_label"],
            "candidate_evaluations": EVALUATIONS_PER_SEARCH,
            "inner_mean_loss": mean_loss,
            "inner_std_loss": float(candidate["std_inner_loss"]),
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
                float(inner_oof_ba) if calibrated else np.nan
            ),
            "threshold_training_scope": (
                "outer_train_inner_oof" if calibrated else "svc_default"
            ),
            "outer_test_used_for_selection": False,
            "runtime_seconds": time.monotonic() - started,
        },
        time.monotonic() - started,
    )


def append_summary_rows(rows: list[dict], filename: str) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    group_columns = ["dataset", "preprocessing_config", "method"]
    value_columns = [
        column for column in METRICS + [
            "C", "gamma", "candidate_evaluations", "runtime_seconds"
        ]
        if column in frame.columns
    ]
    summary = (
        frame.groupby(group_columns)[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(OUTPUT_DIR / filename, index=False)


def write_paired_comparisons(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    keys = ["dataset", "preprocessing_config", "outer_fold"]
    metrics = METRICS
    wide = frame.pivot(index=keys, columns="method", values=metrics)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    methods = ["M1 Standard PSO", "M2 Post-hoc UA", "M3 UA-PSO", "M4 Random"]
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            for metric in metrics:
                wide[f"{metric}_{left}_minus_{right}"] = (
                    wide[f"{metric}_{left}"] - wide[f"{metric}_{right}"]
                )
    wide.reset_index().to_csv(
        OUTPUT_DIR / "paired_method_comparisons.csv", index=False
    )


def write_findings(
    primary_rows: list[dict],
    lambda_rows: list[dict],
    status_rows: list[dict],
) -> None:
    if not primary_rows:
        return
    data = pd.DataFrame(primary_rows)
    lines = [
        "# v2 Findings Summary",
        "",
        "M3 uses mean inner-fold loss plus lambda times sample SD during particle "
        "fitness and best updates. M2 changes only final selection from M1's "
        "shared candidate pool.",
        "",
        "## Primary default-threshold results by dataset",
        "",
        "Means pool four preprocessing variants and five outer folds and are "
        "descriptive only.",
        "",
        "| Dataset | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, method), group in data.groupby(["dataset", "method"]):
        lines.append(
            f"| {dataset} | {method} | "
            + " | ".join(f"{group[m].mean():.4f}" for m in METRICS)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Overall descriptive comparison",
            "",
            "Unweighted means over the 60 dataset/configuration/fold combinations "
            "per method; these observations are overlapping and not independent.",
            "",
            "| Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, group in data.groupby("method"):
        lines.append(
            f"| {method} | "
            + " | ".join(f"{group[metric].mean():.4f}" for metric in METRICS)
            + " |"
        )

    wide = data.pivot(
        index=["dataset", "preprocessing_config", "outer_fold"],
        columns="method",
        values=METRICS,
    )
    lines.extend(
        [
            "",
            "## Paired M3 minus M1 differences",
            "",
            "Mean paired fold/configuration differences; no inferential test is "
            "performed.",
            "",
            "| Dataset | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ loss gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        subset = wide.loc[dataset]
        deltas = {
            metric: (
                subset[(metric, "M3 UA-PSO")]
                - subset[(metric, "M1 Standard PSO")]
            ).mean()
            for metric in METRICS
        }
        lines.append(
            f"| {dataset} | "
            + " | ".join(f"{deltas[metric]:.4f}" for metric in METRICS)
            + " |"
        )

    m1 = data[data["method"] == "M1 Standard PSO"].set_index(
        ["dataset", "preprocessing_config", "outer_fold"]
    )
    m3 = data[data["method"] == "M3 UA-PSO"].set_index(
        ["dataset", "preprocessing_config", "outer_fold"]
    )
    paired_keys = m1.index.intersection(m3.index)
    m1 = m1.loc[paired_keys]
    m3 = m3.loc[paired_keys]
    selected_pair_changed = (
        (m1["C"] - m3["C"]).abs().gt(1e-12)
        | (m1["gamma"] - m3["gamma"]).abs().gt(1e-12)
    )
    absolute_gap_better = int(
        (m3["inner_to_outer_loss_gap"].abs()
         < m1["inner_to_outer_loss_gap"].abs()).sum()
    )
    lines.extend(
        [
            "",
            "## Hypothesis-level observations",
            "",
            f"- **H1 (trajectory/selection change):** M1 and M3 had different "
            f"post-initial particle positions in all 60 matched searches; their "
            f"selected C/gamma differed in {int(selected_pair_changed.sum())} "
            "of 60 outer-fold comparisons.",
            f"- **H2 (predictive improvement):** M3 minus M1 mean ROC-AUC was "
            f"{m3['outer_roc_auc'].mean() - m1['outer_roc_auc'].mean():+.4f}, "
            f"balanced accuracy {m3['balanced_accuracy'].mean() - m1['balanced_accuracy'].mean():+.4f}, "
            f"and F1 {m3['f1_score'].mean() - m1['f1_score'].mean():+.4f} "
            "across the matched pooled rows. Effects varied by dataset.",
            "- **H3 (seed stability):** not tested; only one seed was run.",
            f"- **H4 (generalization gap):** M3's signed mean gap was "
            f"{m3['inner_to_outer_loss_gap'].mean():+.4f} versus "
            f"{m1['inner_to_outer_loss_gap'].mean():+.4f} for M1, but a negative "
            "gap is not automatically better. The absolute gap was smaller for "
            f"M3 in only {absolute_gap_better}/60 matched rows; no consistent "
            "gap reduction is established.",
        ]
    )

    lambda_frame = pd.DataFrame(lambda_rows)
    if not lambda_frame.empty:
        lines.extend(
            [
                "",
                "## M3 lambda sensitivity",
                "",
                "Each nonzero lambda has its own PSO trajectory; lambda 0 reuses "
                "M1. Means pool four configurations and five folds per dataset.",
                "",
                "| Dataset | Lambda | ROC-AUC | Balanced accuracy | F1 | Mean inner loss | Mean inner SD |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (dataset, lam), group in lambda_frame.groupby(["dataset", "lambda"]):
            lines.append(
                f"| {dataset} | {lam:g} | {group['outer_roc_auc'].mean():.4f} | "
                f"{group['balanced_accuracy'].mean():.4f} | "
                f"{group['f1_score'].mean():.4f} | "
                f"{group['inner_mean_loss'].mean():.4f} | "
                f"{group['inner_std_loss'].mean():.4f} |"
            )

    failed = [row for row in status_rows if row["status"] == "failed"]
    measured_elapsed = sum(row.get("runtime_seconds", 0.0) for row in status_rows)
    lines.extend(
        [
            "",
            "## Status and interpretation",
            "",
            f"- Completed configurations: {sum(row['status'] == 'completed' for row in status_rows)}/{len(status_rows)}; failures: {len(failed)}.",
            f"- Total measured elapsed time across configurations: {measured_elapsed:.2f} seconds ({measured_elapsed / 3600:.2f} hours). This includes unusually long pauses recorded within some folds and is not a compute-only estimate.",
            "- One seed was used. No repeated-seed stability conclusion is supported.",
            "- The completed historical run used `TruncatedSVD.random_state=None`; "
            "this source now sets it from `SVD_RANDOM_STATE` for future runs. "
            "Existing SVD results were not regenerated or independently reproduced.",
            "- Outer folds overlap and are not independent; no statistical "
            "significance is claimed.",
            "- This three-dataset/configuration pooled table is descriptive, not "
            "a pooled clinical performance estimate.",
            "- Threshold-calibrated results are separate; calibration uses "
            "outer-training inner OOF scores and inherits the score-scale "
            "mismatch when applied after refitting on all outer-training rows.",
            "- Better accuracy from a larger search is not assumed; assess each "
            "method and dataset separately.",
        ]
    )
    if failed:
        lines.extend(["", "### Failed combinations"])
        for row in failed:
            lines.append(
                f"- {row['dataset']} / {row['preprocessing_config']}: "
                f"{row.get('error', 'See failure_log.txt')}"
            )
    (OUTPUT_DIR / "findings_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if any((OUTPUT_DIR / filename).exists() for filename in RESULT_FILES):
        raise FileExistsError(
            f"Experiment output already exists; refusing to overwrite: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    started = time.monotonic()
    expected_total = len(DATASETS) * len(PREPROCESSING_CONFIGS)
    primary_rows: list[dict] = []
    calibrated_rows: list[dict] = []
    candidate_rows: list[dict] = []
    lambda_rows: list[dict] = []
    runtime_rows: list[dict] = []
    status_rows: list[dict] = []
    error_path = OUTPUT_DIR / "failure_log.txt"
    error_path.write_text("No failures recorded yet.\n", encoding="utf-8")
    config_path = OUTPUT_DIR / "experiment_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "outer_folds": OUTER_FOLDS,
                "inner_folds": INNER_FOLDS,
                "particles": N_PARTICLES,
                "iterations": N_ITERATIONS,
                "evaluations_per_search": EVALUATIONS_PER_SEARCH,
                "log10_C_bounds": [LOWER_BOUND, UPPER_BOUND],
                "log10_gamma_bounds": [LOWER_BOUND, UPPER_BOUND],
                "primary_lambda": PRIMARY_LAMBDA,
                "lambda_sensitivity_values": LAMBDA_VALUES,
                "fitness_sample_std_ddof": 1,
                "pso_coefficients": {
                    "inertia": 0.7, "cognitive": 1.5, "social": 1.5
                },
                "datasets": DATASETS,
                "preprocessing_configs": PREPROCESSING_CONFIGS,
                "pca_variance": PCA_VARIANCE,
                "svd_components": SVD_COMPONENTS,
                "svd_random_state": SVD_RANDOM_STATE,
                "method_search_design": {
                    "M1": "PSO mean loss",
                    "M2": "post-hoc selection from identical M1 history",
                    "M3": "separate penalized-fitness PSO trajectory per lambda",
                    "M4": "48 uniformly random points; not PSO",
                    "matched_initial_positions_and_rng": True,
                },
                "repeatability_study": "not run; one-seed initial experiment",
                "output_directory": str(OUTPUT_DIR.resolve()),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    logging.info("Outputs isolated to %s", OUTPUT_DIR)

    for dataset_index, dataset in enumerate(DATASETS, start=1):
        for config_index, config in enumerate(PREPROCESSING_CONFIGS, start=1):
            status = {
                "dataset": dataset,
                "preprocessing_config": config["name"],
                "status": "in_progress",
                "completed_outer_folds": 0,
                "expected_outer_folds": OUTER_FOLDS,
                "completed_candidate_evaluations": 0,
                "runtime_seconds": 0.0,
                "error": "",
            }
            status_rows.append(status)
            checkpoint(status_rows, "completion_log.csv")
            combination_started = time.monotonic()
            ordinal = (dataset_index - 1) * len(PREPROCESSING_CONFIGS) + config_index
            logging.info("Starting %d/%d: %s / %s", ordinal, expected_total, dataset, config["name"])
            try:
                X, y, categorical_features = get_dataset(dataset)
                numeric_features = [
                    column for column in X.columns
                    if column not in categorical_features
                ]
                outer_cv = StratifiedKFold(
                    n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
                )
                for outer_fold, (train_idx, test_idx) in enumerate(
                    outer_cv.split(X, y), start=1
                ):
                    fold_started = time.monotonic()
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    inner_cv = StratifiedKFold(
                        n_splits=INNER_FOLDS,
                        shuffle=True,
                        random_state=SEED + outer_fold,
                    )
                    inner_splits = list(inner_cv.split(X_train, y_train))

                    lambda_histories: dict[float, list[dict]] = {}
                    lambda_runtimes: dict[float, float] = {}
                    for lam in LAMBDA_VALUES:
                        if lam == 0.0:
                            label = "M1_standard_shared_pool"
                        else:
                            label = f"M3_lambda_{lam:g}"
                        history, search_runtime = run_pso_search(
                            X_train, y_train, numeric_features,
                            categorical_features, config, dataset, outer_fold,
                            lam, label, candidate_rows, inner_splits
                        )
                        lambda_histories[lam] = history
                        lambda_runtimes[lam] = search_runtime
                        status["completed_candidate_evaluations"] += len(history)
                        checkpoint(candidate_rows, "candidate_search_history.csv")
                        runtime_rows.append(
                            {
                                "dataset": dataset,
                                "preprocessing_config": config["name"],
                                "outer_fold": outer_fold,
                                "search_label": label,
                                "method_family": "PSO",
                                "lambda": lam,
                                "candidate_evaluations": len(history),
                                "runtime_seconds": search_runtime,
                            }
                        )

                    random_history, random_runtime = run_random_search(
                        X_train, y_train, numeric_features,
                        categorical_features, config, dataset, outer_fold,
                        candidate_rows, inner_splits
                    )
                    status["completed_candidate_evaluations"] += len(random_history)
                    checkpoint(candidate_rows, "candidate_search_history.csv")
                    runtime_rows.append(
                        {
                            "dataset": dataset,
                            "preprocessing_config": config["name"],
                            "outer_fold": outer_fold,
                            "search_label": "M4_random_search",
                            "method_family": "Random",
                            "lambda": 0.0,
                            "candidate_evaluations": len(random_history),
                            "runtime_seconds": random_runtime,
                        }
                    )

                    m1_pool = lambda_histories[0.0]
                    primary_candidates = {
                        "M1 Standard PSO": select_candidate(m1_pool, 0.0),
                        "M2 Post-hoc UA": select_candidate(m1_pool, PRIMARY_LAMBDA),
                        "M3 UA-PSO": select_candidate(
                            lambda_histories[PRIMARY_LAMBDA], PRIMARY_LAMBDA
                        ),
                        "M4 Random": select_candidate(random_history, 0.0),
                    }
                    selected_ids: dict[
                        tuple[str, str, int, str, int], list[str]
                    ] = {}
                    for method, candidate in primary_candidates.items():
                        key = (
                            dataset, config["name"], outer_fold,
                            candidate["search_label"], candidate["evaluation"],
                        )
                        selected_ids.setdefault(key, []).append(method)
                    for row in candidate_rows:
                        selected = selected_ids.get(
                            (
                                row["dataset"], row["preprocessing_config"],
                                row["outer_fold"], row["search_label"],
                                row["evaluation"],
                            ),
                            [],
                        )
                        if selected:
                            row["selected_methods"] = ";".join(selected)

                    primary_search_runtime = {
                        "M1 Standard PSO": lambda_runtimes[0.0],
                        "M2 Post-hoc UA": lambda_runtimes[0.0],
                        "M3 UA-PSO": lambda_runtimes[PRIMARY_LAMBDA],
                        "M4 Random": random_runtime,
                    }
                    primary_metrics: dict[str, dict] = {}
                    for method, candidate in primary_candidates.items():
                        metrics, eval_runtime = score_candidate(
                            candidate, method, X_train, y_train, X_test, y_test,
                            numeric_features, categorical_features, config,
                            inner_splits, calibrated=False,
                            selection_lambda=(
                                PRIMARY_LAMBDA
                                if method in {"M2 Post-hoc UA", "M3 UA-PSO"}
                                else 0.0
                            )
                        )
                        result = {
                            "dataset": dataset,
                            "preprocessing_config": config["name"],
                            "numeric_imputation": config["numeric_imputation"],
                            "dimensionality_reduction": (
                                config["dimensionality_reduction"] or "none"
                            ),
                            "outer_fold": outer_fold,
                            "seed": SEED,
                            "inner_seed": SEED + outer_fold,
                            "candidate_pool_id": candidate["search_label"],
                            "candidate_pool_evaluations": EVALUATIONS_PER_SEARCH,
                            "search_runtime_seconds": primary_search_runtime[method],
                            "final_evaluation_runtime_seconds": eval_runtime,
                            **metrics,
                        }
                        primary_rows.append(result)
                        primary_metrics[method] = result

                        calibrated_metrics, calibrated_runtime = score_candidate(
                            candidate, method, X_train, y_train, X_test, y_test,
                            numeric_features, categorical_features, config,
                            inner_splits, calibrated=True,
                            selection_lambda=(
                                PRIMARY_LAMBDA
                                if method in {"M2 Post-hoc UA", "M3 UA-PSO"}
                                else 0.0
                            )
                        )
                        calibrated_rows.append(
                            {
                                "dataset": dataset,
                                "preprocessing_config": config["name"],
                                "outer_fold": outer_fold,
                                "seed": SEED,
                                "candidate_pool_id": candidate["search_label"],
                                "candidate_pool_evaluations": EVALUATIONS_PER_SEARCH,
                                "final_evaluation_runtime_seconds": calibrated_runtime,
                                **calibrated_metrics,
                            }
                        )

                    for lam in LAMBDA_VALUES:
                        pool = lambda_histories[lam]
                        selected = (
                            primary_candidates["M1 Standard PSO"]
                            if lam == 0.0
                            else select_candidate(pool, lam)
                        )
                        method_name = (
                            "M1 Standard PSO" if lam == 0.0 else "M3 UA-PSO"
                        )
                        if lam == 0.0:
                            metrics = primary_metrics[method_name]
                            eval_runtime = metrics["final_evaluation_runtime_seconds"]
                        elif lam == PRIMARY_LAMBDA:
                            metrics = primary_metrics["M3 UA-PSO"]
                            eval_runtime = metrics["final_evaluation_runtime_seconds"]
                        else:
                            scored, eval_runtime = score_candidate(
                                selected, method_name, X_train, y_train,
                                X_test, y_test, numeric_features,
                                categorical_features, config, inner_splits,
                                calibrated=False, selection_lambda=lam
                            )
                            metrics = scored
                        lambda_rows.append(
                            {
                                "dataset": dataset,
                                "preprocessing_config": config["name"],
                                "outer_fold": outer_fold,
                                "lambda": lam,
                                "method": method_name,
                                "search_label": selected["search_label"],
                                "selected_evaluation": selected["evaluation"],
                                "C": selected["C"],
                                "gamma": selected["gamma"],
                                "inner_mean_loss": selected["mean_inner_loss"],
                                "inner_std_loss": selected["std_inner_loss"],
                                "fitness": (
                                    selected["mean_inner_loss"]
                                    + lam * selected["std_inner_loss"]
                                ),
                                "outer_roc_auc": metrics["outer_roc_auc"],
                                "balanced_accuracy": metrics["balanced_accuracy"],
                                "sensitivity": metrics["sensitivity"],
                                "specificity": metrics["specificity"],
                                "f1_score": metrics["f1_score"],
                                "inner_to_outer_loss_gap": metrics[
                                    "inner_to_outer_loss_gap"
                                ],
                                "candidate_evaluations": EVALUATIONS_PER_SEARCH,
                                "search_runtime_seconds": lambda_runtimes[lam],
                                "final_evaluation_runtime_seconds": eval_runtime,
                            }
                        )

                    status["completed_outer_folds"] = outer_fold
                    checkpoint(primary_rows, "outer_fold_results.csv")
                    checkpoint(candidate_rows, "candidate_search_history.csv")
                    checkpoint(calibrated_rows, "threshold_calibrated_fold_results.csv")
                    checkpoint(lambda_rows, "lambda_sensitivity.csv")
                    checkpoint(runtime_rows, "runtime_summary.csv")
                    checkpoint(status_rows, "completion_log.csv")
                    logging.info(
                        "Completed %s/%s fold %d in %.2fs; searches=%d evaluations",
                        dataset, config["name"], outer_fold,
                        time.monotonic() - fold_started,
                        status["completed_candidate_evaluations"],
                    )
                status["status"] = "completed"
            except Exception as error:
                status["status"] = "failed"
                status["error"] = f"{type(error).__name__}: {error}"
                with error_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"\n[{dataset} / {config['name']}]\n"
                        f"{traceback.format_exc()}\n"
                    )
                logging.exception("Failed %s / %s", dataset, config["name"])
            finally:
                status["runtime_seconds"] = time.monotonic() - combination_started
                checkpoint(primary_rows, "outer_fold_results.csv")
                checkpoint(candidate_rows, "candidate_search_history.csv")
                checkpoint(calibrated_rows, "threshold_calibrated_fold_results.csv")
                checkpoint(lambda_rows, "lambda_sensitivity.csv")
                checkpoint(runtime_rows, "runtime_summary.csv")
                checkpoint(status_rows, "completion_log.csv")

    checkpoint(primary_rows, "outer_fold_results.csv")
    checkpoint(candidate_rows, "candidate_search_history.csv")
    checkpoint(calibrated_rows, "threshold_calibrated_fold_results.csv")
    checkpoint(lambda_rows, "lambda_sensitivity.csv")
    checkpoint(runtime_rows, "runtime_summary.csv")
    checkpoint(status_rows, "completion_log.csv")
    append_summary_rows(primary_rows, "aggregated_results.csv")
    write_paired_comparisons(primary_rows)
    write_findings(primary_rows, lambda_rows, status_rows)
    logging.info("Experiment finished in %.2fs", time.monotonic() - started)


if __name__ == "__main__":
    main()
