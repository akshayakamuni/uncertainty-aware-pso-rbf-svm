"""Validate lambda-sensitivity experiment configuration and saved outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_lambda_multiseed as runner

LAMBDAS = {0.0, 0.25, 0.5, 1.0, 2.0}
NONZERO_LAMBDAS = {0.25, 0.5, 1.0, 2.0}
METHODS = {"M1 Standard PSO", "M3 UA-PSO"}
METRICS = runner.METRICS
REQUIRED_SEED_FILES = set(runner.OUTPUT_FILES) - {"validation_report.md"}


def validate_static_config() -> list[str]:
    errors = []
    try:
        config = runner.load_config()
    except Exception as error:
        return [f"Cannot load protocol config: {type(error).__name__}: {error}"]
    if config["lambda_values"] != [0.0, 0.25, 0.5, 1.0, 2.0]:
        errors.append("Configured lambda values are not exactly the required list.")
    if config["seeds"] != [42, 43, 44]:
        errors.append("Configured seeds must be exactly 42, 43, and 44.")
    helper = runner.HELPERS
    if helper.EVALUATIONS_PER_SEARCH != 48:
        errors.append("The source PSO budget is not 48 evaluations per search.")
    if helper.LOWER_BOUND != -3 or helper.UPPER_BOUND != 3:
        errors.append("Source hyperparameter bounds differ from [-3, 3].")
    if helper.PRIMARY_LAMBDA != 1.0:
        errors.append("Source primary lambda differs from 1.")
    if helper.SVD_RANDOM_STATE != helper.SEED:
        errors.append("Read-only v2 source does not explicitly seed SVD.")
    return errors


def validate_seed_directory(
    seed_dir: Path, config: dict | None = None, write_report: bool = True
) -> list[str]:
    errors = []
    missing = sorted(
        name for name in REQUIRED_SEED_FILES if not (seed_dir / name).is_file()
    )
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))
        if write_report:
            _write_report(seed_dir, errors)
        return errors

    config = runner.load_config() if config is None else config
    actual = json.loads(
        (seed_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    seed = int(actual.get("seed", -1))
    if seed not in config["seeds"]:
        errors.append(f"Seed {seed} is not in the configured seeds.")
    if actual.get("svd_random_state") != seed:
        errors.append("Recorded SVD random_state does not equal this run's seed.")

    runner.HELPERS.SEED = seed
    runner.HELPERS.SVD_RANDOM_STATE = seed
    svd_pipeline = runner.HELPERS.create_model(
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
    if svd.random_state != seed:
        errors.append("Constructed SVD preprocessing pipeline is not seeded.")

    try:
        results = pd.read_csv(seed_dir / "outer_fold_results.csv")
        calibrated = pd.read_csv(
            seed_dir / "threshold_calibrated_fold_results.csv"
        )
        candidates = pd.read_csv(seed_dir / "candidate_search_history.csv")
        runtime = pd.read_csv(seed_dir / "runtime_summary.csv")
        completion = pd.read_csv(seed_dir / "completion_log.csv")
    except pd.errors.EmptyDataError:
        errors.append("One or more seed CSV files are empty.")
        if write_report:
            _write_report(seed_dir, errors)
        return errors

    expected_results = (
        len(config["datasets"])
        * len(config["preprocessing_configs"])
        * config["outer_folds"]
        * len(LAMBDAS)
    )
    expected_candidates = expected_results * config["evaluations_per_search"]
    if len(results) != expected_results:
        errors.append(f"Expected {expected_results} result rows; found {len(results)}.")
    if len(calibrated) != expected_results:
        errors.append(
            f"Expected {expected_results} calibrated rows; found {len(calibrated)}."
        )
    if len(candidates) != expected_candidates:
        errors.append(
            f"Expected {expected_candidates} candidates; found {len(candidates)}."
        )
    if len(runtime) != expected_results:
        errors.append(f"Expected {expected_results} runtime rows; found {len(runtime)}.")

    result_keys = [
        "seed", "dataset", "preprocessing_config", "outer_fold", "lambda"
    ]
    _validate_paired_result_keys(results, calibrated, config, seed, errors)
    _validate_key_domain(
        runtime,
        result_keys,
        _expected_result_keys(config, seed),
        "runtime results",
        errors,
    )
    if runtime.duplicated(result_keys).any():
        errors.append("Duplicate runtime result keys.")
    candidate_keys = result_keys + ["evaluation"]
    if candidates.duplicated(candidate_keys).any():
        errors.append("Duplicate candidate-history keys.")
    candidate_pool_keys = set(
        map(
            tuple,
            candidates[result_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    expected_key_set = _expected_result_keys(config, seed)
    if candidate_pool_keys != expected_key_set:
        errors.append(
            "Candidate-history pools do not cover the complete expected "
            "dataset/configuration/fold/lambda key domain."
        )

    for frame_name, frame in (("default", results), ("calibrated", calibrated)):
        if set(frame["lambda"].unique()) != LAMBDAS:
            errors.append(f"{frame_name} results do not contain all lambdas.")
        expected_methods = frame["lambda"].map(
            lambda value: "M1 Standard PSO" if float(value) == 0 else "M3 UA-PSO"
        )
        if not frame["method"].eq(expected_methods).all():
            errors.append(f"{frame_name} results have incorrect method labels.")
        if not frame["seed"].eq(seed).all():
            errors.append(f"{frame_name} rows contain an incorrect seed.")
        if not frame["svd_random_state"].eq(seed).all():
            errors.append(f"{frame_name} rows do not record the correct SVD seed.")
        if not frame["candidate_evaluations"].eq(48).all():
            errors.append(f"{frame_name} results do not report 48 evaluations.")
        if not frame["outer_test_used_for_selection"].eq(False).all():
            errors.append(f"Outer test appears to influence {frame_name} selection.")
        if frame_name == "default":
            if not frame["threshold_training_scope"].eq("svc_default").all():
                errors.append("Default output contains calibrated threshold metrics.")
        elif not frame["threshold_training_scope"].eq(
            "outer_train_inner_oof"
        ).all():
            errors.append("Calibrated threshold was not restricted to outer-train OOF.")

    if not calibrated["seed"].eq(seed).all():
        errors.append("Calibrated rows contain the wrong run seed.")
    for frame_name, frame in (("default", results), ("calibrated", calibrated)):
        finite_columns = [
            "outer_roc_auc", "balanced_accuracy", "sensitivity", "specificity",
            "f1_score", "inner_mean_loss", "inner_std_loss",
            "inner_to_outer_loss_gap", "absolute_inner_to_outer_loss_gap",
            "C", "gamma", "candidate_evaluations",
        ]
        if not np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all():
            errors.append(f"{frame_name} outputs contain non-finite numeric values.")
        for column in (
            "outer_roc_auc", "balanced_accuracy", "sensitivity",
            "specificity", "f1_score",
        ):
            if not frame[column].between(0.0, 1.0).all():
                errors.append(f"{frame_name} metric {column} is outside [0, 1].")
        if not np.allclose(
            frame["absolute_inner_to_outer_loss_gap"],
            frame["inner_to_outer_loss_gap"].abs(),
        ):
            errors.append(f"{frame_name} absolute gap is not abs(signed gap).")

    if not candidates["outer_test_used_for_search"].eq(False).all():
        errors.append("Candidate history flags outer-test use during search.")
    if not candidates["lambda"].isin(LAMBDAS).all():
        errors.append("Candidate history contains an unconfigured lambda.")
    for lambda_value, group in candidates.groupby("lambda"):
        expected_label = (
            "M1 Standard PSO" if lambda_value == 0 else "M3 UA-PSO"
        )
        if not group["method"].eq(expected_label).all():
            errors.append(f"Candidate history method does not match lambda={lambda_value}.")
        if not group["fitness"].sub(
            group["mean_inner_loss"]
            + lambda_value * group["std_inner_loss"]
        ).abs().le(1e-12).all():
            errors.append(f"Candidate fitness is incorrect at lambda={lambda_value}.")

    _validate_matched_initial_positions(candidates, errors)
    _validate_selections(results, candidates, errors)
    _validate_runtime(runtime, config, errors)
    _validate_completion(completion, config, seed, errors)

    if write_report:
        _write_report(seed_dir, errors)
    return errors


def _expected_result_keys(config: dict, seed: int) -> set[tuple]:
    return set(
        product(
            [seed],
            config["datasets"],
            [item["name"] for item in config["preprocessing_configs"]],
            range(1, config["outer_folds"] + 1),
            sorted(LAMBDAS),
        )
    )


def _validate_paired_result_keys(
    results: pd.DataFrame,
    calibrated: pd.DataFrame,
    config: dict,
    seed: int,
    errors: list[str],
) -> None:
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold", "lambda"]
    expected = _expected_result_keys(config, seed)
    _validate_key_domain(
        results, keys, expected, "default-threshold results", errors
    )
    _validate_key_domain(
        calibrated,
        keys,
        expected,
        "threshold-calibrated results",
        errors,
    )
    if results.duplicated(keys).any():
        errors.append("Duplicate default-threshold result keys.")
    if calibrated.duplicated(keys).any():
        errors.append("Duplicate calibrated result keys.")
    default_keys = set(
        map(tuple, results[keys].itertuples(index=False, name=None))
    )
    calibrated_keys = set(
        map(tuple, calibrated[keys].itertuples(index=False, name=None))
    )
    if default_keys != calibrated_keys:
        errors.append(
            "Default-threshold and threshold-calibrated result keys do not match."
        )


def _validate_key_domain(
    frame: pd.DataFrame,
    keys: list[str],
    expected: set[tuple],
    label: str,
    errors: list[str],
) -> None:
    actual = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        errors.append(f"{label} are missing {len(missing)} expected keys.")
    if unexpected:
        errors.append(f"{label} contain {len(unexpected)} unexpected keys.")


def _validate_matched_initial_positions(
    candidates: pd.DataFrame, errors: list[str]
) -> None:
    initial = candidates[candidates["iteration"] == 1]
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold", "particle"]
    for _, group in initial.groupby(keys):
        positions = group[["initial_log10_C", "initial_log10_gamma"]]
        if len(positions.drop_duplicates()) != 1 or set(group["lambda"]) != LAMBDAS:
            errors.append("Initial PSO coordinates do not match across all lambdas.")
            return


def _validate_selections(
    results: pd.DataFrame, candidates: pd.DataFrame, errors: list[str]
) -> None:
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold", "lambda"]
    for _, result in results.iterrows():
        mask = pd.Series(True, index=candidates.index)
        for key in keys:
            mask &= candidates[key].eq(result[key])
        pool = candidates[mask]
        if len(pool) != 48:
            errors.append(f"Expected 48 candidates for result key {tuple(result[k] for k in keys)}.")
            continue
        selected = pool[pool["evaluation"].eq(result["selected_evaluation"])]
        if len(selected) != 1:
            errors.append("Selected candidate is absent or duplicated in its pool.")
            continue
        row = selected.iloc[0]
        if not (
            math.isclose(float(row["C"]), float(result["C"]), rel_tol=1e-10)
            and math.isclose(float(row["gamma"]), float(result["gamma"]), rel_tol=1e-10)
        ):
            errors.append("Selected C/gamma differ from candidate history.")
        lam = float(result["lambda"])
        best_score = float(
            (pool["mean_inner_loss"] + lam * pool["std_inner_loss"]).min()
        )
        selected_score = float(
            row["mean_inner_loss"] + lam * row["std_inner_loss"]
        )
        if selected_score > best_score + 1e-12:
            errors.append("Selected candidate does not minimize its lambda criterion.")
        if (
            not math.isclose(float(row["mean_inner_loss"]), float(result["inner_mean_loss"]), abs_tol=1e-12)
            or not math.isclose(float(row["std_inner_loss"]), float(result["inner_std_loss"]), abs_tol=1e-12)
        ):
            errors.append("Selected inner mean/sample SD differ from candidate history.")


def _validate_runtime(
    runtime: pd.DataFrame, config: dict, errors: list[str]
) -> None:
    if not runtime["candidate_evaluations"].eq(48).all():
        errors.append("Runtime rows do not report 48 evaluations.")
    if not np.isfinite(runtime["search_runtime_seconds"].to_numpy(dtype=float)).all():
        errors.append("Runtime rows contain non-finite timings.")
    if not runtime["lambda"].isin(LAMBDAS).all():
        errors.append("Runtime rows contain an unconfigured lambda.")


def _validate_completion(
    completion: pd.DataFrame, config: dict, seed: int, errors: list[str]
) -> None:
    expected = len(config["datasets"]) * len(config["preprocessing_configs"])
    if len(completion) != expected:
        errors.append(f"Expected {expected} completion rows; got {len(completion)}.")
        return
    if not completion["seed"].eq(seed).all():
        errors.append("Completion log contains an incorrect seed.")
    if not completion["completed_outer_folds"].eq(config["outer_folds"]).all():
        errors.append("A dataset/configuration is missing outer folds.")
    required_searches = config["outer_folds"] * len(config["lambda_values"])
    if not completion["completed_lambda_searches"].eq(required_searches).all():
        errors.append("A dataset/configuration is missing lambda searches.")
    if not completion["status"].isin(
        ["validation_pending", "completed", "validation_failed"]
    ).all():
        errors.append("A dataset/configuration failed or is still incomplete.")


def _write_report(seed_dir: Path, errors: list[str]) -> None:
    lines = [
        f"# Lambda sensitivity validation — {seed_dir.name}",
        "",
        f"Result: {'PASS' if not errors else 'FAIL'}",
        "",
    ]
    if errors:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append(
            "All expected lambdas, methods, candidate budgets, selected "
            "candidates, matched initial positions, SVD seeds, leakage flags, "
            "threshold scopes, and completion keys passed."
        )
    (seed_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_aggregates(config: dict) -> list[str]:
    errors = []
    required = set(runner.AGGREGATE_FILES)
    missing = sorted(name for name in required if not (EXPERIMENT_DIR / name).is_file())
    if missing:
        return ["Missing aggregate files: " + ", ".join(missing)]
    default = pd.read_csv(EXPERIMENT_DIR / "paired_lambda_fold_differences.csv")
    calibrated = pd.read_csv(
        EXPERIMENT_DIR / "paired_lambda_calibrated_fold_differences.csv"
    )
    keys = ["seed", "dataset", "preprocessing_config", "outer_fold", "lambda"]
    expected_rows = (
        len(config["seeds"])
        * len(config["datasets"])
        * len(config["preprocessing_configs"])
        * config["outer_folds"]
        * len(NONZERO_LAMBDAS)
    )
    for label, frame in (("default", default), ("calibrated", calibrated)):
        if len(frame) != expected_rows:
            errors.append(f"{label} paired row count is {len(frame)}; expected {expected_rows}.")
        if frame.duplicated(keys).any():
            errors.append(f"{label} paired output has duplicate keys.")
    notes = (EXPERIMENT_DIR / "README.md").read_text(encoding="utf-8")
    if not _documents_dependence_limitations(notes):
        errors.append("README omits dependence limitations.")
    return errors


def _documents_dependence_limitations(notes: str) -> bool:
    lowered = notes.lower()
    return "independent" in lowered and any(
        phrase in lowered for phrase in ("not treat", "not independent")
    )


def promote_seed(seed: int, config: dict) -> list[str]:
    seed_dir = runner.seed_output_path(seed)
    errors = validate_seed_directory(seed_dir, config, write_report=True)
    if errors:
        return errors
    completion_path = seed_dir / "completion_log.csv"
    completion = pd.read_csv(completion_path)
    completion["status"] = "completed"
    completion.to_csv(completion_path, index=False)
    config_path = seed_dir / "experiment_config.json"
    actual = json.loads(config_path.read_text(encoding="utf-8"))
    actual["validation_status"] = "passed"
    config_path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    return validate_seed_directory(seed_dir, config, write_report=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--promote-seed", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    errors = validate_static_config()
    config = runner.load_config()
    if args.promote_seed:
        if args.seed is None:
            errors.append("--promote-seed requires --seed.")
        elif not errors:
            errors.extend(promote_seed(args.seed, config))
    elif args.aggregate and not errors:
        runner.aggregate_completed_seeds(config)
        errors.extend(validate_aggregates(config))
    elif args.seed is not None and not errors:
        errors.extend(
            validate_seed_directory(
                runner.seed_output_path(args.seed), config, write_report=True
            )
        )
    elif not args.check_config and not errors:
        for seed in config["seeds"]:
            errors.extend(
                validate_seed_directory(
                    runner.seed_output_path(seed), config, write_report=True
                )
            )
        if not errors:
            errors.extend(validate_aggregates(config))
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
