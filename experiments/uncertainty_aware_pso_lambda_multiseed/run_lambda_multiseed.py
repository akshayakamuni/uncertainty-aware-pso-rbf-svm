"""Run nested-CV multi-seed PSO searches across uncertainty penalty values."""

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
CONFIG_PATH = EXPERIMENT_DIR / "lambda_multiseed_config.json"
METRICS = [
    "outer_roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
    "inner_mean_loss",
    "inner_std_loss",
    "inner_to_outer_loss_gap",
    "absolute_inner_to_outer_loss_gap",
]
OUTPUT_FILES = (
    "experiment_config.json",
    "outer_fold_results.csv",
    "threshold_calibrated_fold_results.csv",
    "candidate_search_history.csv",
    "runtime_summary.csv",
    "completion_log.csv",
    "experiment.log",
    "failure_log.txt",
    "validation_report.md",
)
AGGREGATE_FILES = (
    "paired_lambda_fold_differences.csv",
    "paired_lambda_calibrated_fold_differences.csv",
    "paired_lambda_summary_by_seed.csv",
    "paired_lambda_calibrated_summary_by_seed.csv",
    "paired_lambda_summary_across_seeds.csv",
    "paired_lambda_calibrated_summary_across_seeds.csv",
    "findings_summary.md",
)

sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_v2_helpers():
    if not V2_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Read-only v2 helper runner missing: {V2_RUNNER_PATH}")
    name = "_uncertainty_aware_pso_lambda_multiseed_helpers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V2_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load v2 helper runner from {V2_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_v2_helpers()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["lambda_values"] != [0.0, 0.25, 0.5, 1.0, 2.0]:
        raise ValueError("Lambda values must be exactly 0, 0.25, 0.5, 1, and 2.")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Configured seeds must be distinct.")
    if (
        config["outer_folds"] != 5
        or config["inner_folds"] != 5
        or config["particles"] != 8
        or config["iterations"] != 6
        or config["evaluations_per_search"] != 48
    ):
        raise ValueError("Configured CV protocol or PSO budget differs from v2.")
    expected_bounds = [HELPERS.LOWER_BOUND, HELPERS.UPPER_BOUND]
    if (
        config["log10_C_bounds"] != expected_bounds
        or config["log10_gamma_bounds"] != expected_bounds
    ):
        raise ValueError("Configured search bounds differ from v2.")
    if config["datasets"] != HELPERS.DATASETS:
        raise ValueError("Configured datasets differ from v2.")
    if config["preprocessing_configs"] != HELPERS.PREPROCESSING_CONFIGS:
        raise ValueError("Configured preprocessing differs from v2.")
    if (
        config["pca_variance"] != HELPERS.PCA_VARIANCE
        or config["svd_components"] != HELPERS.SVD_COMPONENTS
    ):
        raise ValueError("Configured PCA/SVD settings differ from v2.")
    return config


def seed_output_path(seed: int) -> Path:
    return EXPERIMENT_DIR / f"seed_{seed}"


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def preflight_outputs(seeds: list[int]) -> None:
    for seed in seeds:
        require_absent(seed_output_path(seed))
    for name in AGGREGATE_FILES:
        path = EXPERIMENT_DIR / name
        if name == "findings_summary.md":
            if not path.is_file() or "**Status: Not run.**" not in path.read_text(
                encoding="utf-8"
            ):
                raise FileExistsError(
                    f"Refusing to overwrite non-placeholder findings summary: {path}"
                )
        else:
            require_absent(path)


def expected_search_count(config: dict[str, Any]) -> int:
    return (
        len(config["datasets"])
        * len(config["preprocessing_configs"])
        * config["outer_folds"]
        * len(config["lambda_values"])
    )


def expected_candidate_count(config: dict[str, Any]) -> int:
    return expected_search_count(config) * config["evaluations_per_search"]


def build_nested_splits(X, y, seed: int, config: dict[str, Any]):
    outer = StratifiedKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=seed
    )
    plans = []
    for fold, (train_indices, test_indices) in enumerate(outer.split(X, y), 1):
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        inner = StratifiedKFold(
            n_splits=config["inner_folds"],
            shuffle=True,
            random_state=seed + fold,
        )
        plans.append(
            (
                fold,
                train_indices,
                test_indices,
                list(inner.split(X_train, y_train)),
            )
        )
    return plans


def assert_matched_initializations(histories: dict[float, list[dict]]) -> None:
    if set(histories) != {0.0, 0.25, 0.5, 1.0, 2.0}:
        raise AssertionError("Matched initialization check requires all five lambdas.")
    initial_by_lambda = {}
    for lambda_value, history in histories.items():
        initial_by_lambda[lambda_value] = {
            int(row["particle"]): (
                float(row["initial_log10_C"]),
                float(row["initial_log10_gamma"]),
            )
            for row in history
            if int(row["iteration"]) == 1
        }
    reference = initial_by_lambda[0.0]
    if len(reference) != HELPERS.N_PARTICLES or any(
        positions != reference for positions in initial_by_lambda.values()
    ):
        raise AssertionError("Initial PSO particle positions differ across lambdas.")


def assert_candidate_budget(history: list[dict], expected: int) -> None:
    if len(history) != expected:
        raise AssertionError(f"Expected {expected} candidates; got {len(history)}.")
    evaluations = [int(row["evaluation"]) for row in history]
    if sorted(evaluations) != list(range(1, expected + 1)):
        raise AssertionError("Candidate evaluation IDs are incomplete or duplicated.")


def select_lambda_candidate(
    history: list[dict], lambda_value: float
) -> dict:
    if lambda_value not in {0.0, 0.25, 0.5, 1.0, 2.0}:
        raise ValueError(f"Unconfigured lambda: {lambda_value}")
    return HELPERS.select_candidate(history, lambda_value)


def write_checkpoint(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure_logging(seed_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(seed_dir / "experiment.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _result_row(
    seed: int,
    dataset: str,
    preprocessing: dict,
    fold: int,
    lambda_value: float,
    candidate: dict,
    scores: dict,
    search_runtime: float,
    evaluation_runtime: float,
    calibrated: bool,
) -> dict:
    signed_gap = float(scores["inner_to_outer_loss_gap"])
    return {
        "seed": seed,
        "inner_seed": seed + fold,
        "pso_seed": seed + fold,
        "svd_random_state": seed,
        "dataset": dataset,
        "preprocessing_config": preprocessing["name"],
        "outer_fold": fold,
        "lambda": lambda_value,
        "method": "M1 Standard PSO" if lambda_value == 0 else "M3 UA-PSO",
        "candidate_pool_id": candidate["search_label"],
        "selected_evaluation": int(candidate["evaluation"]),
        "C": float(candidate["C"]),
        "gamma": float(candidate["gamma"]),
        "candidate_evaluations": int(scores["candidate_evaluations"]),
        "search_runtime_seconds": float(search_runtime),
        "final_evaluation_runtime_seconds": float(evaluation_runtime),
        "inner_mean_loss": float(scores["inner_mean_loss"]),
        "inner_std_loss": float(scores["inner_std_loss"]),
        "outer_roc_auc": float(scores["outer_roc_auc"]),
        "balanced_accuracy": float(scores["balanced_accuracy"]),
        "sensitivity": float(scores["sensitivity"]),
        "specificity": float(scores["specificity"]),
        "f1_score": float(scores["f1_score"]),
        "inner_to_outer_loss_gap": signed_gap,
        "absolute_inner_to_outer_loss_gap": abs(signed_gap),
        "threshold": float(scores["threshold"]),
        "threshold_training_scope": scores["threshold_training_scope"],
        "inner_oof_balanced_accuracy_at_threshold": scores[
            "inner_oof_balanced_accuracy_at_threshold"
        ],
        "outer_test_used_for_selection": bool(scores["outer_test_used_for_selection"]),
        "calibrated": calibrated,
    }


def run_seed(seed: int, config: dict[str, Any]) -> bool:
    started = time.monotonic()
    seed_dir = seed_output_path(seed)
    require_absent(seed_dir)
    seed_dir.mkdir(parents=False)
    configure_logging(seed_dir)
    HELPERS.SEED = seed
    HELPERS.SVD_RANDOM_STATE = seed
    actual_config = {
        **config,
        "seed": seed,
        "outer_random_state": seed,
        "inner_random_states": {
            str(fold): seed + fold
            for fold in range(1, config["outer_folds"] + 1)
        },
        "pso_random_states": {
            str(fold): seed + fold
            for fold in range(1, config["outer_folds"] + 1)
        },
        "svd_random_state": seed,
        "source_v2_runner": str(V2_RUNNER_PATH.resolve()),
        "output_directory": str(seed_dir.resolve()),
        "validation_status": "in_progress",
    }
    (seed_dir / "experiment_config.json").write_text(
        json.dumps(actual_config, indent=2) + "\n", encoding="utf-8"
    )
    (seed_dir / "failure_log.txt").write_text(
        "No failures recorded.\n", encoding="utf-8"
    )

    default_rows: list[dict] = []
    calibrated_rows: list[dict] = []
    candidate_rows: list[dict] = []
    runtime_rows: list[dict] = []
    completion_rows: list[dict] = []
    failures: list[str] = []

    for dataset in config["datasets"]:
        for preprocessing in config["preprocessing_configs"]:
            status = {
                "seed": seed,
                "dataset": dataset,
                "preprocessing_config": preprocessing["name"],
                "status": "in_progress",
                "completed_outer_folds": 0,
                "expected_outer_folds": config["outer_folds"],
                "completed_lambda_searches": 0,
                "expected_lambda_searches": (
                    config["outer_folds"] * len(config["lambda_values"])
                ),
                "error": "",
            }
            completion_rows.append(status)
            write_checkpoint(completion_rows, seed_dir / "completion_log.csv")
            try:
                X, y, categorical = HELPERS.get_dataset(dataset)
                numeric = [column for column in X.columns if column not in categorical]
                split_plan = build_nested_splits(X, y, seed, config)
                for fold, train_idx, test_idx, inner_splits in split_plan:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    histories: dict[float, list[dict]] = {}
                    runtimes: dict[float, float] = {}

                    for lambda_value in config["lambda_values"]:
                        label = (
                            "M1_standard"
                            if lambda_value == 0
                            else f"M3_lambda_{lambda_value:g}"
                        )
                        history, search_runtime = HELPERS.run_pso_search(
                            X_train,
                            y_train,
                            numeric,
                            categorical,
                            preprocessing,
                            dataset,
                            fold,
                            lambda_value,
                            f"{label}_seed_{seed}",
                            candidate_rows,
                            inner_splits,
                        )
                        assert_candidate_budget(
                            history, config["evaluations_per_search"]
                        )
                        histories[lambda_value] = history
                        runtimes[lambda_value] = search_runtime
                        for row in history:
                            row["seed"] = seed
                            row["lambda"] = lambda_value
                            row["method"] = (
                                "M1 Standard PSO"
                                if lambda_value == 0
                                else "M3 UA-PSO"
                            )
                            row["inner_seed"] = seed + fold
                            row["pso_seed"] = seed + fold
                            row["svd_random_state"] = seed
                        runtime_rows.append(
                            {
                                "seed": seed,
                                "dataset": dataset,
                                "preprocessing_config": preprocessing["name"],
                                "outer_fold": fold,
                                "lambda": lambda_value,
                                "method": (
                                    "M1 Standard PSO"
                                    if lambda_value == 0
                                    else "M3 UA-PSO"
                                ),
                                "candidate_evaluations": len(history),
                                "search_runtime_seconds": search_runtime,
                            }
                        )

                    assert_matched_initializations(histories)
                    for lambda_value in config["lambda_values"]:
                        candidate = select_lambda_candidate(
                            histories[lambda_value], lambda_value
                        )
                        method = (
                            "M1 Standard PSO"
                            if lambda_value == 0
                            else "M3 UA-PSO"
                        )
                        default_scores, default_runtime = HELPERS.score_candidate(
                            candidate,
                            method,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            numeric,
                            categorical,
                            preprocessing,
                            inner_splits,
                            calibrated=False,
                            selection_lambda=lambda_value,
                        )
                        default_rows.append(
                            _result_row(
                                seed, dataset, preprocessing, fold, lambda_value,
                                candidate, default_scores, runtimes[lambda_value],
                                default_runtime, calibrated=False,
                            )
                        )
                        calibrated_scores, calibrated_runtime = HELPERS.score_candidate(
                            candidate,
                            method,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            numeric,
                            categorical,
                            preprocessing,
                            inner_splits,
                            calibrated=True,
                            selection_lambda=lambda_value,
                        )
                        calibrated_rows.append(
                            _result_row(
                                seed, dataset, preprocessing, fold, lambda_value,
                                candidate, calibrated_scores, runtimes[lambda_value],
                                calibrated_runtime, calibrated=True,
                            )
                        )
                    status["completed_outer_folds"] = fold
                    status["completed_lambda_searches"] += len(config["lambda_values"])
                    write_checkpoint(default_rows, seed_dir / "outer_fold_results.csv")
                    write_checkpoint(
                        calibrated_rows,
                        seed_dir / "threshold_calibrated_fold_results.csv",
                    )
                    write_checkpoint(
                        candidate_rows, seed_dir / "candidate_search_history.csv"
                    )
                    write_checkpoint(runtime_rows, seed_dir / "runtime_summary.csv")
                    write_checkpoint(completion_rows, seed_dir / "completion_log.csv")
                status["status"] = "validation_pending"
            except Exception as error:
                status["status"] = "failed"
                status["error"] = f"{type(error).__name__}: {error}"
                failures.append(
                    f"{dataset}/{preprocessing['name']}:\n{traceback.format_exc()}"
                )
                logging.exception(
                    "Failed seed %s, %s / %s", seed, dataset, preprocessing["name"]
                )
            finally:
                write_checkpoint(completion_rows, seed_dir / "completion_log.csv")

    for rows, name in (
        (default_rows, "outer_fold_results.csv"),
        (calibrated_rows, "threshold_calibrated_fold_results.csv"),
        (candidate_rows, "candidate_search_history.csv"),
        (runtime_rows, "runtime_summary.csv"),
    ):
        write_checkpoint(rows, seed_dir / name)
    if failures:
        (seed_dir / "failure_log.txt").write_text(
            "\n".join(failures), encoding="utf-8"
        )

    from validate_lambda_multiseed import validate_seed_directory

    validation_errors = validate_seed_directory(seed_dir, config, write_report=True)
    actual_config["elapsed_runtime_seconds"] = time.monotonic() - started
    if failures or validation_errors:
        actual_config["validation_status"] = "failed"
        (seed_dir / "experiment_config.json").write_text(
            json.dumps(actual_config, indent=2) + "\n", encoding="utf-8"
        )
        for row in completion_rows:
            if row["status"] != "failed":
                row["status"] = "validation_failed"
        write_checkpoint(completion_rows, seed_dir / "completion_log.csv")
        logging.error("Seed %s failed validation; outputs are preserved.", seed)
        return False

    actual_config["validation_status"] = "passed"
    (seed_dir / "experiment_config.json").write_text(
        json.dumps(actual_config, indent=2) + "\n", encoding="utf-8"
    )
    for row in completion_rows:
        row["status"] = "completed"
    write_checkpoint(completion_rows, seed_dir / "completion_log.csv")
    logging.info("Seed %s validated and marked complete.", seed)
    return True


def _paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold"]
    base = frame[frame["lambda"] == 0].set_index(keys)
    comparisons = []
    for lambda_value in (0.25, 0.5, 1.0, 2.0):
        current = frame[frame["lambda"] == lambda_value].set_index(keys)
        joined = current[METRICS].join(
            base[METRICS], lsuffix="_lambda", rsuffix="_lambda_0", how="inner"
        ).reset_index()
        joined["lambda"] = lambda_value
        for metric in METRICS:
            joined[f"{metric}_lambda_minus_0"] = (
                joined[f"{metric}_lambda"] - joined[f"{metric}_lambda_0"]
            )
        comparisons.append(joined)
    if not comparisons:
        raise ValueError("No nonzero lambda comparisons available.")
    return pd.concat(comparisons, ignore_index=True)


def _summarize_paired(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = [f"{metric}_lambda_minus_0" for metric in METRICS]
    by_seed = (
        paired.groupby(
            ["seed", "dataset", "preprocessing_config", "lambda"]
        )[deltas]
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
        ["seed", "dataset", "preprocessing_config", "lambda"], as_index=False
    )[deltas].mean()
    across_seeds = (
        seed_means.groupby(["dataset", "preprocessing_config", "lambda"])[deltas]
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


def write_findings_summary(
    default_summary: pd.DataFrame,
    calibrated_summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    metrics = [
        "outer_roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "absolute_inner_to_outer_loss_gap",
    ]
    lines = [
        "# Multi-seed lambda sensitivity findings",
        "",
        "Lambda=0 is the standard mean-inner-loss PSO objective (M1). For each "
        "nonzero lambda, M3 applies mean loss + lambda × sample SD during PSO "
        "fitness and personal/global-best updates, then selects from that "
        "lambda-specific candidate pool. All comparisons are paired against "
        "lambda=0 by seed, dataset, preprocessing configuration, and outer fold.",
        "",
        "## Default-threshold paired differences",
        "",
        "Entries are means of the five outer-fold paired differences, averaged "
        "across the three seed-level means. Counts are descriptive; no "
        "significance test is performed.",
        "",
        "| Dataset | Preprocessing | Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ absolute gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in default_summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['preprocessing_config']} | "
            f"{row['lambda']:g} | "
            + " | ".join(
                f"{row[f'{metric}_lambda_minus_0_mean']:+.4f}"
                for metric in metrics
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Threshold-calibrated paired differences",
            "",
            "Threshold calibration is separate and uses only outer-training "
            "inner out-of-fold decision scores. The outer-test fold is used "
            "only for evaluation.",
            "",
            "| Dataset | Preprocessing | Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in calibrated_summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['preprocessing_config']} | "
            f"{row['lambda']:g} | "
            + " | ".join(
                f"{row[f'{metric}_lambda_minus_0_mean']:+.4f}"
                for metric in metrics[:5]
            )
            + " |"
        )

    runtime_values = []
    for seed in config["seeds"]:
        run_config = json.loads(
            (seed_output_path(seed) / "experiment_config.json").read_text(
                encoding="utf-8"
            )
        )
        runtime_values.append(
            (seed, float(run_config["elapsed_runtime_seconds"]))
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| Seed | Elapsed seconds | Elapsed hours |",
            "|---:|---:|---:|",
        ]
    )
    for seed, seconds in runtime_values:
        lines.append(f"| {seed} | {seconds:.2f} | {seconds / 3600:.2f} |")
    total = sum(seconds for _, seconds in runtime_values)
    lines.append(f"| Total sequential | {total:.2f} | {total / 3600:.2f} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Outer folds overlap; folds and seeds on the same datasets are "
            "not independent observations.",
            "- These are descriptive comparisons; no statistical significance "
            "is claimed.",
            "- This experiment does not establish universal superiority, "
            "reduced overtuning, clinical benefit, or novelty.",
            "- Only the three declared seeds and three datasets were tested.",
            "- Lambda changes the PSO objective and its best updates; it is "
            "not a final-selection-only post-hoc comparison.",
            "- Default-threshold and calibrated outputs are stored separately.",
        ]
    )
    (EXPERIMENT_DIR / "findings_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def aggregate_completed_seeds(config: dict[str, Any]) -> None:
    for name in AGGREGATE_FILES:
        path = EXPERIMENT_DIR / name
        if name == "findings_summary.md":
            if not path.is_file() or "**Status: Not run.**" not in path.read_text(
                encoding="utf-8"
            ):
                raise FileExistsError(
                    f"Refusing to replace non-placeholder findings summary: {path}"
                )
        else:
            require_absent(path)
    from validate_lambda_multiseed import validate_seed_directory

    validated_seed_dirs = []
    for seed in config["seeds"]:
        seed_dir = seed_output_path(seed)
        validation_errors = validate_seed_directory(
            seed_dir, config, write_report=True
        )
        if validation_errors:
            raise RuntimeError(
                f"Seed {seed} failed fresh validation before aggregation: "
                + "; ".join(validation_errors)
            )
        run_config = json.loads(
            (seed_dir / "experiment_config.json").read_text(encoding="utf-8")
        )
        if run_config.get("validation_status") != "passed":
            raise RuntimeError(f"Seed {seed} has not passed validation.")
        validated_seed_dirs.append(seed_dir)

    default_frames, calibrated_frames = [], []
    for seed_dir in validated_seed_dirs:
        default_frames.append(pd.read_csv(seed_dir / "outer_fold_results.csv"))
        calibrated_frames.append(
            pd.read_csv(seed_dir / "threshold_calibrated_fold_results.csv")
        )

    default = pd.concat(default_frames, ignore_index=True)
    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    paired_default = _paired_differences(default)
    paired_calibrated = _paired_differences(calibrated)
    default_by_seed, default_across = _summarize_paired(paired_default)
    calibrated_by_seed, calibrated_across = _summarize_paired(paired_calibrated)
    paired_default.to_csv(
        EXPERIMENT_DIR / "paired_lambda_fold_differences.csv", index=False
    )
    paired_calibrated.to_csv(
        EXPERIMENT_DIR / "paired_lambda_calibrated_fold_differences.csv",
        index=False,
    )
    default_by_seed.to_csv(
        EXPERIMENT_DIR / "paired_lambda_summary_by_seed.csv", index=False
    )
    calibrated_by_seed.to_csv(
        EXPERIMENT_DIR / "paired_lambda_calibrated_summary_by_seed.csv",
        index=False,
    )
    default_across.to_csv(
        EXPERIMENT_DIR / "paired_lambda_summary_across_seeds.csv", index=False
    )
    calibrated_across.to_csv(
        EXPERIMENT_DIR / "paired_lambda_calibrated_summary_across_seeds.csv",
        index=False,
    )
    write_findings_summary(default_across, calibrated_across, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Configured seed subset; aggregation requires all configured seeds.",
    )
    args = parser.parse_args()
    config = load_config()
    seeds = config["seeds"] if args.seeds is None else args.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify one or more distinct configured seeds.")
    invalid = sorted(set(seeds) - set(config["seeds"]))
    if invalid:
        raise ValueError(f"Unconfigured seeds are not allowed: {invalid}")

    preflight_outputs(seeds)
    all_seeds_validated = True
    for seed in seeds:
        if not run_seed(seed, config):
            all_seeds_validated = False
    if set(seeds) == set(config["seeds"]) and all_seeds_validated:
        aggregate_completed_seeds(config)
    elif not all_seeds_validated:
        raise SystemExit(
            "One or more seeds failed; partial outputs are preserved and "
            "aggregates were not generated."
        )


if __name__ == "__main__":
    main()
