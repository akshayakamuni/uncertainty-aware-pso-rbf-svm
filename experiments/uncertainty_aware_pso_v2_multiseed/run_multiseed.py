"""Run matched M1/M3 nested-CV experiments in isolated per-seed directories."""

from __future__ import annotations

import argparse
import importlib.util
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
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
V2_RUNNER_PATH = (
    EXPERIMENT_DIR.parent / "uncertainty_aware_pso_v2" / "run_experiment.py"
)
CONFIG_PATH = EXPERIMENT_DIR / "multiseed_config.json"
METRICS = [
    "outer_roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
    "inner_to_outer_loss_gap",
]
OUTPUT_FILES = (
    "experiment_config.json",
    "outer_fold_results.csv",
    "candidate_search_history.csv",
    "runtime_summary.csv",
    "completion_log.csv",
    "experiment.log",
    "failure_log.txt",
    "validation_report.md",
)


def load_v2_helpers():
    sys.dont_write_bytecode = True
    if not V2_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Read-only v2 helper runner missing: {V2_RUNNER_PATH}")
    module_name = "_uncertainty_aware_pso_v2_multiseed_helpers"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, V2_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load v2 helper runner from {V2_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_v2_helpers()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["methods"] != ["M1 Standard PSO", "M3 UA-PSO"]:
        raise ValueError("This runner is restricted to M1 Standard PSO and M3 UA-PSO.")
    if config["primary_lambda"] != 1.0:
        raise ValueError("This runner requires lambda=1.0.")
    if (
        config["outer_folds"] != 5
        or config["inner_folds"] != 5
        or config["particles"] != 8
        or config["iterations"] != 6
        or config["evaluations_per_search"] != 48
    ):
        raise ValueError("Configured nested CV or candidate budget differs from protocol.")
    if (
        config["log10_C_bounds"] != [HELPERS.LOWER_BOUND, HELPERS.UPPER_BOUND]
        or config["log10_gamma_bounds"] != [HELPERS.LOWER_BOUND, HELPERS.UPPER_BOUND]
    ):
        raise ValueError("Configured C/gamma bounds differ from the v2 search bounds.")
    if config["datasets"] != HELPERS.DATASETS:
        raise ValueError("Configured datasets differ from the v2 experiment.")
    if config["preprocessing_configs"] != HELPERS.PREPROCESSING_CONFIGS:
        raise ValueError("Configured preprocessing differs from the v2 experiment.")
    return config


def seed_output_path(seed: int) -> Path:
    return EXPERIMENT_DIR / f"seed_{seed}"


def require_output_path_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing seed output directory: {path}"
        )


def build_nested_splits(X, y, seed: int, config: dict[str, Any]):
    outer = StratifiedKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=seed
    )
    split_plan = []
    for outer_fold, (train_indices, test_indices) in enumerate(outer.split(X, y), 1):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        inner = StratifiedKFold(
            n_splits=config["inner_folds"],
            shuffle=True,
            random_state=seed + outer_fold,
        )
        inner_splits = list(inner.split(X_train, y_train))
        split_plan.append((outer_fold, train_indices, test_indices, inner_splits))
    return split_plan


def assert_matched_initializations(
    m1_history: list[dict], m3_history: list[dict]
) -> None:
    def initial_positions(history: list[dict]) -> dict[tuple[int, int], tuple[float, float]]:
        return {
            (int(row["iteration"]), int(row["particle"])): (
                float(row["initial_log10_C"]),
                float(row["initial_log10_gamma"]),
            )
            for row in history
        }

    m1_positions = initial_positions(m1_history)
    m3_positions = initial_positions(m3_history)
    if not m1_positions or m1_positions != m3_positions:
        raise AssertionError("M1 and M3 initial particle coordinates do not match.")


def assert_candidate_budget(
    history: list[dict], expected_evaluations: int
) -> None:
    if len(history) != expected_evaluations:
        raise AssertionError(
            f"Expected {expected_evaluations} candidate evaluations; found {len(history)}."
        )
    evaluations = [int(row["evaluation"]) for row in history]
    if sorted(evaluations) != list(range(1, expected_evaluations + 1)):
        raise AssertionError("Candidate evaluation IDs are incomplete or duplicated.")


def select_method_candidates(
    m1_pool: list[dict], m3_pool: list[dict], lambda_value: float = 1.0
) -> tuple[dict, dict]:
    if lambda_value != 1.0:
        raise ValueError("M3 selection is fixed at lambda=1.0.")
    return (
        HELPERS.select_candidate(m1_pool, 0.0),
        HELPERS.select_candidate(m3_pool, lambda_value),
    )


def run_matched_pso_searches(
    X_train,
    y_train,
    numeric_features: list[str],
    categorical_features: list[str],
    preprocessing: dict[str, Any],
    dataset: str,
    outer_fold: int,
    seed: int,
    candidate_rows: list[dict],
    inner_splits: list,
    evaluations_per_search: int,
) -> tuple[list[dict], float, list[dict], float]:
    HELPERS.SEED = seed
    HELPERS.SVD_RANDOM_STATE = seed
    m1_history, m1_runtime = HELPERS.run_pso_search(
        X_train, y_train, numeric_features, categorical_features,
        preprocessing, dataset, outer_fold, 0.0,
        f"M1_standard_seed_{seed}", candidate_rows, inner_splits,
    )
    m3_history, m3_runtime = HELPERS.run_pso_search(
        X_train, y_train, numeric_features, categorical_features,
        preprocessing, dataset, outer_fold, 1.0,
        f"M3_UA_lambda_1_seed_{seed}", candidate_rows, inner_splits,
    )
    assert_candidate_budget(m1_history, evaluations_per_search)
    assert_candidate_budget(m3_history, evaluations_per_search)
    assert_matched_initializations(m1_history, m3_history)
    return m1_history, m1_runtime, m3_history, m3_runtime


def build_paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold"]
    required_methods = {"M1 Standard PSO", "M3 UA-PSO"}
    if not required_methods.issubset(set(results["method"].unique())):
        raise ValueError("Paired results require both M1 and M3.")
    if results.duplicated(keys + ["method"]).any():
        raise ValueError("Result keys are not unique by seed/config/fold/method.")
    wide = results.pivot(index=keys, columns="method", values=METRICS)
    paired = wide.reset_index()
    paired.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in paired.columns
    ]
    for metric in METRICS:
        paired[f"{metric}_M3_minus_M1"] = (
            paired[f"{metric}_M3 UA-PSO"]
            - paired[f"{metric}_M1 Standard PSO"]
        )
    return paired


def summarize_paired_differences(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_columns = [f"{metric}_M3_minus_M1" for metric in METRICS]
    by_seed = (
        paired.groupby(["seed", "dataset", "preprocessing_config"])[delta_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_seed.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in by_seed.columns
    ]
    seed_means = paired.groupby(
        ["seed", "dataset", "preprocessing_config"], as_index=False
    )[delta_columns].mean()
    across_seeds = (
        seed_means.groupby(["dataset", "preprocessing_config"])[delta_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    across_seeds.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in across_seeds.columns
    ]
    return by_seed, across_seeds


def write_csv_checkpoint(rows: list[dict], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def configure_seed_logging(seed_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(seed_dir / "experiment.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def run_seed(seed: int, config: dict[str, Any]) -> bool:
    seed_started = time.monotonic()
    seed_dir = seed_output_path(seed)
    require_output_path_absent(seed_dir)
    seed_dir.mkdir(parents=False)
    configure_seed_logging(seed_dir)

    HELPERS.SEED = seed
    HELPERS.SVD_RANDOM_STATE = seed
    run_config = {
        **config,
        "seed": seed,
        "svd_random_state": seed,
        "inner_seed_rule": "seed + 1-based outer_fold",
        "pso_seed_rule": "seed + 1-based outer_fold; same fresh RNG per method",
        "output_directory": str(seed_dir.resolve()),
        "source_v2_runner": str(V2_RUNNER_PATH.resolve()),
        "validation_status": "in_progress",
    }
    (seed_dir / "experiment_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    (seed_dir / "failure_log.txt").write_text(
        "No failures recorded.\n", encoding="utf-8"
    )

    result_rows: list[dict] = []
    candidate_rows: list[dict] = []
    runtime_rows: list[dict] = []
    completion_rows: list[dict] = []
    errors: list[str] = []

    for dataset in config["datasets"]:
        for preprocessing in config["preprocessing_configs"]:
            status = {
                "seed": seed,
                "dataset": dataset,
                "preprocessing_config": preprocessing["name"],
                "status": "in_progress",
                "completed_outer_folds": 0,
                "expected_outer_folds": config["outer_folds"],
                "error": "",
            }
            completion_rows.append(status)
            write_csv_checkpoint(completion_rows, seed_dir / "completion_log.csv")
            try:
                X, y, categorical_features = HELPERS.get_dataset(dataset)
                numeric_features = [
                    column for column in X.columns
                    if column not in categorical_features
                ]
                split_plan = build_nested_splits(X, y, seed, config)
                for outer_fold, train_idx, test_idx, inner_splits in split_plan:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    (
                        m1_history,
                        m1_search_runtime,
                        m3_history,
                        m3_search_runtime,
                    ) = run_matched_pso_searches(
                        X_train,
                        y_train,
                        numeric_features,
                        categorical_features,
                        preprocessing,
                        dataset,
                        outer_fold,
                        seed,
                        candidate_rows,
                        inner_splits,
                        config["evaluations_per_search"],
                    )

                    for method, history, search_runtime, lambda_value in (
                        ("M1 Standard PSO", m1_history, m1_search_runtime, 0.0),
                        ("M3 UA-PSO", m3_history, m3_search_runtime, 1.0),
                    ):
                        for row in history:
                            row["seed"] = seed
                            row["method"] = method
                            row["inner_seed"] = seed + outer_fold
                            row["svd_random_state"] = seed
                        runtime_rows.append(
                            {
                                "seed": seed,
                                "dataset": dataset,
                                "preprocessing_config": preprocessing["name"],
                                "outer_fold": outer_fold,
                                "method": method,
                                "lambda": lambda_value,
                                "candidate_evaluations": len(history),
                                "search_runtime_seconds": search_runtime,
                            }
                        )

                    selected_m1, selected_m3 = select_method_candidates(
                        m1_history, m3_history
                    )
                    for method, candidate in (
                        ("M1 Standard PSO", selected_m1),
                        ("M3 UA-PSO", selected_m3),
                    ):
                        metrics, evaluation_runtime = HELPERS.score_candidate(
                            candidate,
                            method,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            numeric_features,
                            categorical_features,
                            preprocessing,
                            inner_splits,
                            calibrated=False,
                            selection_lambda=0.0 if method.startswith("M1") else 1.0,
                        )
                        result_rows.append(
                            {
                                "seed": seed,
                                "inner_seed": seed + outer_fold,
                                "dataset": dataset,
                                "preprocessing_config": preprocessing["name"],
                                "outer_fold": outer_fold,
                                "method": method,
                                "candidate_pool_id": candidate["search_label"],
                                "selected_evaluation": candidate["evaluation"],
                                "search_runtime_seconds": (
                                    m1_search_runtime
                                    if method.startswith("M1")
                                    else m3_search_runtime
                                ),
                                "final_evaluation_runtime_seconds": evaluation_runtime,
                                "svd_random_state": seed,
                                **metrics,
                            }
                        )
                    status["completed_outer_folds"] = outer_fold
                    write_csv_checkpoint(result_rows, seed_dir / "outer_fold_results.csv")
                    write_csv_checkpoint(candidate_rows, seed_dir / "candidate_search_history.csv")
                    write_csv_checkpoint(runtime_rows, seed_dir / "runtime_summary.csv")
                    write_csv_checkpoint(completion_rows, seed_dir / "completion_log.csv")

                status["status"] = "validation_pending"
            except Exception as error:
                status["status"] = "failed"
                status["error"] = f"{type(error).__name__}: {error}"
                error_text = traceback.format_exc()
                errors.append(f"{dataset}/{preprocessing['name']}: {error_text}")
                logging.exception(
                    "Failed seed %s, %s / %s", seed, dataset, preprocessing["name"]
                )
            finally:
                write_csv_checkpoint(completion_rows, seed_dir / "completion_log.csv")

    for filename, rows in (
        ("outer_fold_results.csv", result_rows),
        ("candidate_search_history.csv", candidate_rows),
        ("runtime_summary.csv", runtime_rows),
    ):
        write_csv_checkpoint(rows, seed_dir / filename)
    if errors:
        (seed_dir / "failure_log.txt").write_text(
            "\n".join(errors), encoding="utf-8"
        )

    from validate_multiseed import validate_seed_directory

    validation_errors = validate_seed_directory(seed_dir, config, write_report=True)
    if errors or validation_errors:
        run_config["validation_status"] = "failed"
        run_config["elapsed_runtime_seconds"] = time.monotonic() - seed_started
        (seed_dir / "experiment_config.json").write_text(
            json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
        )
        for status in completion_rows:
            if status["status"] != "failed":
                status["status"] = "validation_failed"
        write_csv_checkpoint(completion_rows, seed_dir / "completion_log.csv")
        logging.error(
            "Seed %s did not pass validation; it will not be marked complete.", seed
        )
        return False

    run_config["validation_status"] = "passed"
    run_config["elapsed_runtime_seconds"] = time.monotonic() - seed_started
    (seed_dir / "experiment_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    for status in completion_rows:
        status["status"] = "completed"
    write_csv_checkpoint(completion_rows, seed_dir / "completion_log.csv")
    logging.info("Seed %s passed validation and is complete.", seed)
    return True


def aggregate_completed_seeds(config: dict[str, Any]) -> None:
    expected_outputs = [
        "paired_fold_differences.csv",
        "paired_summary_by_seed.csv",
        "paired_summary_across_seeds.csv",
        "aggregation_notes.md",
        "findings_summary.md",
    ]
    existing = [name for name in expected_outputs if (EXPERIMENT_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing aggregate outputs: " + ", ".join(existing)
        )
    seed_frames = []
    for seed in config["seeds"]:
        seed_dir = seed_output_path(seed)
        if not (seed_dir / "experiment_config.json").is_file():
            return
        seed_config = json.loads(
            (seed_dir / "experiment_config.json").read_text(encoding="utf-8")
        )
        if seed_config.get("validation_status") != "passed":
            return
        seed_frames.append(pd.read_csv(seed_dir / "outer_fold_results.csv"))
    if len(seed_frames) != len(config["seeds"]):
        return
    results = pd.concat(seed_frames, ignore_index=True)
    paired = build_paired_differences(results)
    by_seed, across_seeds = summarize_paired_differences(paired)
    paired.to_csv(EXPERIMENT_DIR / "paired_fold_differences.csv", index=False)
    by_seed.to_csv(EXPERIMENT_DIR / "paired_summary_by_seed.csv", index=False)
    across_seeds.to_csv(
        EXPERIMENT_DIR / "paired_summary_across_seeds.csv", index=False
    )
    (EXPERIMENT_DIR / "aggregation_notes.md").write_text(
        "# Multi-seed paired summaries\n\n"
        "The fold-level table pairs M1 and M3 by seed, dataset, preprocessing "
        "configuration, and outer fold. Per-seed summaries average these "
        "matched fold differences. Across-seed summaries describe variation "
        "in the seed-level means. Outer folds overlap and seeds reuse the same "
        "datasets; rows are not independent samples. No statistical "
        "significance is claimed.\n",
        encoding="utf-8",
    )
    write_findings_summary(
        results, paired, by_seed, across_seeds, config
    )


def write_findings_summary(
    results: pd.DataFrame,
    paired: pd.DataFrame,
    by_seed: pd.DataFrame,
    across_seeds: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    report_metrics = [
        "outer_roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    methods = ["M1 Standard PSO", "M3 UA-PSO"]
    lines = [
        "# Multi-seed M1 vs M3 findings",
        "",
        "M1 selects from PSO candidates by mean inner-fold loss. M3 runs PSO "
        "with mean loss + lambda times sample SD as fitness and uses lambda=1. "
        "M3-minus-M1 differences are paired by seed, dataset, preprocessing "
        "configuration, and outer fold.",
        "",
        "## Mean outer-fold metrics by dataset and preprocessing",
        "",
        "| Dataset | Preprocessing | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, preprocessing), group in results.groupby(
        ["dataset", "preprocessing_config"]
    ):
        for method in methods:
            method_rows = group[group["method"] == method]
            lines.append(
                f"| {dataset} | {preprocessing} | {method} | "
                + " | ".join(
                    f"{method_rows[metric].mean():.4f}"
                    for metric in report_metrics
                )
                + " |"
            )

    delta_columns = [f"{metric}_M3_minus_M1" for metric in METRICS]
    lines.extend(
        [
            "",
            "## Paired M3 minus M1 differences by seed, dataset, and preprocessing",
            "",
            "Values below are means of the five matched outer-fold differences "
            "for each seed/dataset/preprocessing combination. The fold-level "
            "paired table is in `paired_fold_differences.csv`.",
            "",
            "| Seed | Dataset | Preprocessing | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_seed.iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['dataset']} | "
            f"{row['preprocessing_config']} | "
            + " | ".join(
                f"{row[f'{metric}_M3_minus_M1_mean']:+.4f}"
                for metric in report_metrics
            )
            + " |"
        )

    consistency_rows = []
    for metric in report_metrics:
        values = by_seed[f"{metric}_M3_minus_M1_mean"]
        consistency_rows.append(
            (
                metric,
                int((values > 0).sum()),
                int((values < 0).sum()),
                int((values == 0).sum()),
            )
        )
    lines.extend(
        [
            "",
            "## Direction consistency",
            "",
            "Counts are over the 36 seed × dataset × preprocessing mean "
            "contrasts (not independent inferential observations).",
            "",
            "| Metric | M3 higher | M3 lower | Equal | Improvement consistent in all groups? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric, positive, negative, zero in consistency_rows:
        lines.append(
            f"| {metric} | {positive} | {negative} | {zero} | "
            f"{'Yes' if positive == len(by_seed) else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Across-seed variation of paired means",
            "",
            "Mean and sample SD are computed over the three seed-level means "
            "for each dataset/preprocessing pair.",
            "",
            "| Dataset | Preprocessing | Metric | Mean M3−M1 | SD across seeds |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in across_seeds.iterrows():
        for metric in report_metrics:
            lines.append(
                f"| {row['dataset']} | {row['preprocessing_config']} | "
                f"{metric} | "
                f"{row[f'{metric}_M3_minus_M1_mean']:+.4f} | "
                f"{row[f'{metric}_M3_minus_M1_std']:.4f} |"
            )

    seed_runtime = []
    for seed in config["seeds"]:
        seed_config = json.loads(
            (seed_output_path(seed) / "experiment_config.json").read_text(
                encoding="utf-8"
            )
        )
        seed_runtime.append((seed, float(seed_config["elapsed_runtime_seconds"])))
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Seed | Elapsed seconds | Elapsed hours |",
            "|---:|---:|---:|",
        ]
    )
    for seed, elapsed in seed_runtime:
        lines.append(f"| {seed} | {elapsed:.2f} | {elapsed / 3600:.2f} |")
    total_elapsed = sum(elapsed for _, elapsed in seed_runtime)
    lines.append(
        f"| Total sequential elapsed | {total_elapsed:.2f} | "
        f"{total_elapsed / 3600:.2f} |"
    )

    lines.extend(
        [
            "",
            "## Validation note",
            "",
            "The first in-process validator used exact winner-evaluation ID "
            "matching. It rejected seeds 43 and 44 where near-tied recorded "
            "losses differed only below the CSV precision used by the runner. "
            "The validator was corrected to allow a selected score within "
            "1e-12 of the best recorded criterion; all three saved seed runs "
            "were revalidated successfully. No candidate searches were rerun, "
            "and seeds 43/44 were marked complete only after that revalidation.",
            "",
            "## Limitations",
            "",
            "- Outer folds overlap; the fold-level comparisons are not independent.",
            "- The three seeds reuse the same underlying datasets; do not treat "
            "seed/fold rows as independent observations.",
            "- These summaries are descriptive; no statistical significance "
            "is claimed.",
            "- SVD is explicitly seeded per run. Historical v2 outputs were "
            "not modified or regenerated by this experiment.",
            "- Runtime is specific to this execution environment.",
        ]
    )
    (EXPERIMENT_DIR / "findings_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Subset of configured seeds to run; defaults to all configured seeds.",
    )
    args = parser.parse_args()
    config = load_config()
    seeds = config["seeds"] if args.seeds is None else args.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify one or more distinct configured seeds.")
    unexpected = sorted(set(seeds) - set(config["seeds"]))
    if unexpected:
        raise ValueError(f"Unconfigured seeds are not allowed: {unexpected}")
    for seed in seeds:
        if not run_seed(seed, config):
            logging.error("Seed %s failed; preserving its output and continuing.", seed)
    aggregate_completed_seeds(config)


if __name__ == "__main__":
    main()
