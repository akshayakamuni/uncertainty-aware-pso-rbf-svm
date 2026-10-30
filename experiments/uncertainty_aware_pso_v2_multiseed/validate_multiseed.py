"""Validate per-seed outputs and paired multi-seed summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_multiseed

DATASETS = {"diabetes", "breast_cancer", "heart_disease"}
PREPROCESSING = {"baseline_median", "mean_imputation", "pca_median", "svd_median"}
METHODS = {"M1 Standard PSO", "M3 UA-PSO"}
REQUIRED_SEED_FILES = {
    "experiment_config.json",
    "outer_fold_results.csv",
    "candidate_search_history.csv",
    "runtime_summary.csv",
    "completion_log.csv",
    "experiment.log",
    "failure_log.txt",
}


def validate_static_config() -> list[str]:
    errors = []
    config = run_multiseed.load_config()
    if config["seeds"] != [42, 43, 44]:
        errors.append("Configured seeds must be exactly 42, 43, and 44.")
    if config["methods"] != ["M1 Standard PSO", "M3 UA-PSO"]:
        errors.append("Configuration contains a method outside M1 and M3.")
    if config["primary_lambda"] != 1.0:
        errors.append("Primary lambda must equal 1.")
    if config["outer_folds"] != 5 or config["inner_folds"] != 5:
        errors.append("Nested CV must use five outer and five inner folds.")
    if config["particles"] != 8 or config["iterations"] != 6:
        errors.append("Search budget must be 8 particles by 6 iterations.")
    if config["evaluations_per_search"] != 48:
        errors.append("Candidate budget must equal 48 per search.")
    if (
        config["log10_C_bounds"] != [
            run_multiseed.HELPERS.LOWER_BOUND,
            run_multiseed.HELPERS.UPPER_BOUND,
        ]
        or config["log10_gamma_bounds"] != [
            run_multiseed.HELPERS.LOWER_BOUND,
            run_multiseed.HELPERS.UPPER_BOUND,
        ]
    ):
        errors.append("Configured search bounds differ from the v2 protocol.")
    if config["datasets"] != run_multiseed.HELPERS.DATASETS:
        errors.append("Configured datasets differ from the v2 protocol.")
    if config["preprocessing_configs"] != run_multiseed.HELPERS.PREPROCESSING_CONFIGS:
        errors.append("Configured preprocessing differs from the v2 protocol.")
    helper = run_multiseed.HELPERS
    if helper.SVD_RANDOM_STATE != helper.SEED:
        errors.append("Read-only v2 helper is not configured to seed SVD.")
    svd_pipeline = helper.create_model(
        1.0,
        1.0,
        ["feature"],
        [],
        {
            "numeric_imputation": "median",
            "dimensionality_reduction": "svd",
        },
    )
    svd = svd_pipeline.named_steps["preprocessing"].named_steps["reduction"]
    if svd.random_state != helper.SVD_RANDOM_STATE:
        errors.append("Constructed SVD preprocessing pipeline lacks an explicit seed.")
    return errors


def _validate_candidate_selection(
    results: pd.DataFrame, candidates: pd.DataFrame, errors: list[str]
) -> None:
    group_keys = [
        "seed", "dataset", "preprocessing_config", "outer_fold", "method"
    ]
    for _, result in results.iterrows():
        pool = candidates[
            (candidates["seed"] == result["seed"])
            & (candidates["dataset"] == result["dataset"])
            & (candidates["preprocessing_config"] == result["preprocessing_config"])
            & (candidates["outer_fold"] == result["outer_fold"])
            & (candidates["method"] == result["method"])
        ]
        if len(pool) != 48:
            errors.append(f"Candidate pool has {len(pool)} rows for {tuple(result[k] for k in group_keys)}.")
            continue
        selected = pool[pool["evaluation"] == result["selected_evaluation"]]
        if len(selected) != 1:
            errors.append("Selected candidate does not uniquely belong to its recorded pool.")
            continue
        if not (
            math.isclose(float(selected.iloc[0]["C"]), float(result["C"]))
            and math.isclose(float(selected.iloc[0]["gamma"]), float(result["gamma"]))
        ):
            errors.append("Selected C/gamma do not match the recorded candidate.")
        criterion_lambda = 0.0 if result["method"] == "M1 Standard PSO" else 1.0
        expected = min(
            pool.to_dict("records"),
            key=lambda row: (
                float(row["mean_inner_loss"])
                + criterion_lambda * float(row["std_inner_loss"]),
                int(row["evaluation"]),
            ),
        )
        selected_score = (
            float(selected.iloc[0]["mean_inner_loss"])
            + criterion_lambda * float(selected.iloc[0]["std_inner_loss"])
        )
        expected_score = (
            float(expected["mean_inner_loss"])
            + criterion_lambda * float(expected["std_inner_loss"])
        )
        if selected_score > expected_score + 1e-12:
            errors.append(
                f"Selection criterion mismatch for {tuple(result[k] for k in group_keys)}."
            )


def validate_seed_directory(
    seed_dir: Path, base_config: dict | None = None, write_report: bool = True
) -> list[str]:
    errors = []
    missing = sorted(name for name in REQUIRED_SEED_FILES if not (seed_dir / name).is_file())
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))
        if write_report:
            write_validation_report(seed_dir, errors)
        return errors
    run_config = json.loads(
        (seed_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    config = run_multiseed.load_config() if base_config is None else base_config
    seed = int(run_config["seed"])
    if seed not in config["seeds"]:
        errors.append(f"Seed {seed} is not in the configured seed list.")
    if run_config.get("svd_random_state") != seed:
        errors.append("Recorded TruncatedSVD random_state does not equal this run's seed.")
    if run_config.get("methods") != sorted(config["methods"]):
        errors.append("Per-seed configuration does not restrict methods to M1 and M3.")
    try:
        results = pd.read_csv(seed_dir / "outer_fold_results.csv")
        candidates = pd.read_csv(seed_dir / "candidate_search_history.csv")
        runtime = pd.read_csv(seed_dir / "runtime_summary.csv")
        completion = pd.read_csv(seed_dir / "completion_log.csv")
    except pd.errors.EmptyDataError:
        errors.append("One or more output CSV files are empty.")
        if write_report:
            write_validation_report(seed_dir, errors)
        return errors

    expected_result_rows = (
        len(DATASETS) * len(PREPROCESSING) * config["outer_folds"] * len(METHODS)
    )
    expected_candidate_rows = expected_result_rows * config["evaluations_per_search"]
    if len(results) != expected_result_rows:
        errors.append(f"Expected {expected_result_rows} result rows; found {len(results)}.")
    if len(candidates) != expected_candidate_rows:
        errors.append(
            f"Expected {expected_candidate_rows} candidate rows; found {len(candidates)}."
        )
    if len(runtime) != expected_result_rows:
        errors.append(
            f"Expected {expected_result_rows} runtime rows; found {len(runtime)}."
        )
    if set(results["dataset"]) != DATASETS or set(results["preprocessing_config"]) != PREPROCESSING:
        errors.append("Result dataset or preprocessing coverage is incomplete.")
    if set(results["method"]) != METHODS or set(candidates["method"]) != METHODS:
        errors.append("Results must contain exactly M1 Standard PSO and M3 UA-PSO.")
    if not results["seed"].eq(seed).all() or not candidates["seed"].eq(seed).all():
        errors.append("Per-seed outputs contain rows with a mismatched seed.")
    if not results["svd_random_state"].eq(seed).all() or not candidates[
        "svd_random_state"
    ].eq(seed).all():
        errors.append("Per-seed output rows do not record the run's SVD seed.")
    if not results["inner_seed"].eq(results["seed"] + results["outer_fold"]).all():
        errors.append("Inner split seeds do not follow the recorded seed rule.")
    result_keys = ["seed", "dataset", "preprocessing_config", "outer_fold", "method"]
    if results.duplicated(result_keys).any():
        errors.append("Duplicate fold-result keys found.")
    if not results.groupby(result_keys[:-1])["method"].agg(
        lambda values: set(values) == METHODS and len(values) == 2
    ).all():
        errors.append("Each matched fold must contain exactly one M1 and one M3 row.")
    candidate_keys = [
        "seed", "dataset", "preprocessing_config", "outer_fold",
        "method", "evaluation",
    ]
    if candidates.duplicated(candidate_keys).any():
        errors.append("Duplicate candidate-history keys found.")
    if not candidates["outer_test_used_for_search"].eq(False).all():
        errors.append("Candidate search indicates outer-test data was used.")
    if not candidates["lambda"].isin([0.0, 1.0]).all():
        errors.append("Candidate history includes a method outside M1/M3 lambda values.")

    numeric_result_columns = [
        "outer_roc_auc", "balanced_accuracy", "sensitivity", "specificity",
        "f1_score", "inner_to_outer_loss_gap", "C", "gamma",
    ]
    if not np.isfinite(results[numeric_result_columns].to_numpy(dtype=float)).all():
        errors.append("Result metrics or selected parameters contain non-finite values.")
    for metric in numeric_result_columns[:5]:
        if not results[metric].between(0.0, 1.0).all():
            errors.append(f"Metric {metric} is outside [0, 1].")
    if not results["candidate_evaluations"].eq(48).all():
        errors.append("Result rows do not report 48 candidate evaluations.")
    if not runtime["candidate_evaluations"].eq(48).all():
        errors.append("Runtime table does not report 48 candidates per search.")
    if not np.isfinite(runtime["search_runtime_seconds"].to_numpy(dtype=float)).all():
        errors.append("Search runtimes contain non-finite values.")

    init_keys = [
        "seed", "dataset", "preprocessing_config", "outer_fold",
        "iteration", "particle",
    ]
    init_columns = ["initial_log10_C", "initial_log10_gamma"]
    initial = candidates[candidates["iteration"] == 1]
    m1_initial = initial[initial["method"] == "M1 Standard PSO"].set_index(init_keys)[init_columns]
    m3_initial = initial[initial["method"] == "M3 UA-PSO"].set_index(init_keys)[init_columns]
    if not m1_initial.equals(m3_initial):
        errors.append("M1 and M3 initial particle coordinates are not identical.")
    _validate_candidate_selection(results, candidates, errors)

    if len(completion) != len(DATASETS) * len(PREPROCESSING):
        errors.append("Completion log does not include every dataset/configuration.")
    elif not (
        completion["status"].isin(
            ["completed", "in_progress", "validation_pending", "validation_failed"]
        ).all()
        and completion["completed_outer_folds"].eq(config["outer_folds"]).all()
    ):
        errors.append("At least one dataset/configuration did not finish all outer folds.")

    if write_report:
        write_validation_report(seed_dir, errors)
    return errors


def write_validation_report(seed_dir: Path, errors: list[str]) -> None:
    lines = [
        f"# Validation Report — {seed_dir.name}",
        "",
        f"Result: {'PASS' if not errors else 'FAIL'}",
        "",
    ]
    if errors:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append(
            "All expected rows, methods, candidate budgets, selected candidates, "
            "seeded SVD settings, matched initial positions, and output keys passed."
        )
    (seed_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_aggregates() -> list[str]:
    errors = []
    names = [
        "paired_fold_differences.csv",
        "paired_summary_by_seed.csv",
        "paired_summary_across_seeds.csv",
        "aggregation_notes.md",
        "findings_summary.md",
    ]
    missing = [name for name in names if not (EXPERIMENT_DIR / name).is_file()]
    if missing:
        return ["Missing aggregate outputs: " + ", ".join(missing)]
    paired = pd.read_csv(EXPERIMENT_DIR / names[0])
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold"]
    if paired.duplicated(keys).any():
        errors.append("Duplicate paired comparison keys.")
    expected_rows = len([42, 43, 44]) * len(DATASETS) * len(PREPROCESSING) * 5
    if len(paired) != expected_rows:
        errors.append(f"Expected {expected_rows} paired rows; found {len(paired)}.")
    for metric in run_multiseed.METRICS:
        for method_column in (f"{metric}_M1 Standard PSO", f"{metric}_M3 UA-PSO"):
            if method_column not in paired:
                errors.append(f"Paired table lacks {method_column}.")
    notes = (EXPERIMENT_DIR / "aggregation_notes.md").read_text(encoding="utf-8")
    if "not independent" not in notes or "No statistical significance" not in notes:
        errors.append("Aggregation notes omit the dependence/significance caveat.")
    findings = (EXPERIMENT_DIR / "findings_summary.md").read_text(
        encoding="utf-8"
    )
    if "Direction consistency" not in findings or "Limitations" not in findings:
        errors.append("Findings summary lacks consistency or limitations sections.")
    return errors


def promote_seed_after_validation(seed: int, config: dict) -> list[str]:
    seed_dir = run_multiseed.seed_output_path(seed)
    errors = validate_seed_directory(seed_dir, config, write_report=True)
    if errors:
        return errors

    completion_path = seed_dir / "completion_log.csv"
    completion = pd.read_csv(completion_path)
    if (
        not completion["completed_outer_folds"].eq(config["outer_folds"]).all()
        or not completion["status"].isin(
            ["completed", "validation_pending", "validation_failed"]
        ).all()
    ):
        errors.append(
            "Cannot promote seed: a configuration did not finish all outer folds."
        )
        write_validation_report(seed_dir, errors)
        return errors

    completion["status"] = "completed"
    completion.to_csv(completion_path, index=False)
    config_path = seed_dir / "experiment_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    run_config["validation_status"] = "passed"
    config_path.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    return validate_seed_directory(seed_dir, config, write_report=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--seed", type=int, help="Validate one seed directory.")
    parser.add_argument(
        "--promote-seed",
        action="store_true",
        help="After successful revalidation, mark a fully completed seed as complete.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Create paired summaries and findings from all validated seeds.",
    )
    args = parser.parse_args()
    errors = validate_static_config()
    if args.promote_seed and args.seed is None:
        errors.append("--promote-seed requires --seed.")
    if args.promote_seed and not errors:
        errors.extend(
            promote_seed_after_validation(
                args.seed, run_multiseed.load_config()
            )
        )
    elif args.seed is not None and not errors:
        errors.extend(
            validate_seed_directory(
                run_multiseed.seed_output_path(args.seed), write_report=True
            )
        )
    elif args.aggregate and not errors:
        config = run_multiseed.load_config()
        run_multiseed.aggregate_completed_seeds(config)
        errors.extend(validate_aggregates())
    elif not errors and not args.check_config:
        config = run_multiseed.load_config()
        for seed in config["seeds"]:
            errors.extend(
                validate_seed_directory(
                    run_multiseed.seed_output_path(seed), config, write_report=True
                )
            )
        if not errors:
            errors.extend(validate_aggregates())
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
