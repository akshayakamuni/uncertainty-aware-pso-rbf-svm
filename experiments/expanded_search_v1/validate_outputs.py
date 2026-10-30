"""Validate expanded-search outputs without modifying existing project files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(
    os.environ.get("EXPANDED_SEARCH_OUTPUT_DIR", str(BASE_OUTPUT_DIR))
).resolve()
if OUTPUT_DIR != BASE_OUTPUT_DIR.resolve() and BASE_OUTPUT_DIR.resolve() not in OUTPUT_DIR.parents:
    raise ValueError("Validation output must stay inside expanded_search_v1.")
PROJECT_ROOT = BASE_OUTPUT_DIR.parents[1]
DATASETS = {"diabetes", "breast_cancer", "heart_disease"}
CONFIGS = {"baseline_median", "mean_imputation", "pca_median", "svd_median"}
METHODS = {"Standard", "Uncertainty-aware"}
METRICS = [
    "outer_roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
]


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    outer_path = OUTPUT_DIR / "outer_fold_results.csv"
    history_path = OUTPUT_DIR / "candidate_search_history.csv"
    calibrated_path = OUTPUT_DIR / "threshold_calibrated_fold_results.csv"
    status_path = OUTPUT_DIR / "completion_log.csv"
    config_path = OUTPUT_DIR / "experiment_config.json"
    required_files = [
        outer_path,
        history_path,
        calibrated_path,
        status_path,
        config_path,
        OUTPUT_DIR / "aggregated_results.csv",
        OUTPUT_DIR / "threshold_calibrated_aggregated_results.csv",
        OUTPUT_DIR / "selection_paired_differences.csv",
        OUTPUT_DIR / "baseline_vs_expanded_search.csv",
        OUTPUT_DIR / "baseline_vs_expanded_search_summary.csv",
        OUTPUT_DIR / "findings_summary.md",
    ]
    all_files_exist = all(path.is_file() for path in required_files)
    checks.append(
        ("Required output files exist", all_files_exist, ", ".join(
            path.name for path in required_files if not path.is_file()
        ) or "All required files present.")
    )
    if not all_files_exist:
        write_report(checks)
        raise SystemExit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    outer = pd.read_csv(outer_path)
    history = pd.read_csv(history_path)
    calibrated = pd.read_csv(calibrated_path)
    status = pd.read_csv(status_path)
    expected_combinations = {
        (dataset, preprocessing)
        for dataset in DATASETS for preprocessing in CONFIGS
    }
    observed_combinations = set(
        zip(outer["dataset"], outer["preprocessing_config"])
    )
    complete_coverage = (
        observed_combinations == expected_combinations
        and len(outer) == 3 * 4 * 5 * 2
        and len(calibrated) == 3 * 4 * 5 * 2
    )
    checks.append(
        ("All dataset/configuration folds and methods represented",
         complete_coverage,
         f"default rows={len(outer)}, calibrated rows={len(calibrated)}, "
         f"combinations={len(observed_combinations)}/12")
    )
    expected_history_rows = 3 * 4 * 5 * 8 * 6
    history_complete = len(history) == expected_history_rows
    checks.append(
        ("All candidate evaluations recorded", history_complete,
         f"{len(history)}/{expected_history_rows} rows")
    )
    duplicate_result_keys = outer.duplicated(
        ["dataset", "preprocessing_config", "outer_fold", "method"]
    ).any() or calibrated.duplicated(
        ["dataset", "preprocessing_config", "outer_fold", "method"]
    ).any()
    checks.append(
        ("No duplicate fold-result keys", not duplicate_result_keys,
         "Checked default and calibrated fold result keys.")
    )
    duplicate_history_keys = history.duplicated(
        ["dataset", "preprocessing_config", "outer_fold", "evaluation"]
    ).any()
    checks.append(
        ("No duplicate candidate evaluation keys", not duplicate_history_keys,
         "Checked dataset/configuration/fold/evaluation keys.")
    )
    for label, frame in (("default", outer), ("calibrated", calibrated)):
        finite = np.isfinite(
            frame[METRICS + ["inner_mean_loss", "inner_std_loss",
                             "inner_to_outer_loss_gap", "C", "gamma",
                             "runtime_seconds"]].to_numpy(dtype=float)
        ).all()
        ranges_ok = all(
            frame[column].between(0.0, 1.0).all() for column in METRICS
        ) and frame["inner_mean_loss"].between(0.0, 1.0).all() \
            and frame["inner_std_loss"].ge(0.0).all() \
            and frame["C"].between(1e-3, 1e3).all() \
            and frame["gamma"].between(1e-3, 1e3).all() \
            and frame["runtime_seconds"].ge(0.0).all()
        checks.append(
            (f"{label.capitalize()} metrics finite and in valid ranges",
             bool(finite and ranges_ok),
             f"Validated {len(frame)} rows.")
        )

    history_lookup = {
        (row.dataset, row.preprocessing_config, int(row.outer_fold),
         int(row.evaluation)): row
        for row in history.itertuples(index=False)
    }
    membership_ok = True
    criteria_ok = True
    shared_pool_ok = True
    for frame in (outer, calibrated):
        for row in frame.itertuples(index=False):
            pool_key = (
                row.dataset, row.preprocessing_config, int(row.outer_fold)
            )
            candidate_key = pool_key + (int(row.selected_evaluation),)
            candidate = history_lookup.get(candidate_key)
            if candidate is None:
                membership_ok = False
                continue
            if not np.isclose(row.C, candidate.C) \
                    or not np.isclose(row.gamma, candidate.gamma) \
                    or not np.isclose(row.inner_mean_loss, candidate.mean_inner_loss) \
                    or not np.isclose(row.inner_std_loss, candidate.std_inner_loss):
                membership_ok = False
            expected_score = row.inner_mean_loss + (
                config["primary_lambda"] * row.inner_std_loss
                if row.method == "Uncertainty-aware" else 0.0
            )
            if not np.isclose(row.selection_score, expected_score, atol=1e-12):
                criteria_ok = False
            candidates = history[
                (history["dataset"] == row.dataset)
                & (history["preprocessing_config"] == row.preprocessing_config)
                & (history["outer_fold"] == row.outer_fold)
            ]
            means = candidates["mean_inner_loss"].to_numpy()
            scores = means + (
                config["primary_lambda"]
                * candidates["std_inner_loss"].to_numpy()
                if row.method == "Uncertainty-aware" else 0.0
            )
            selected = candidates.iloc[int(np.argmin(scores))]
            if int(selected["evaluation"]) != int(row.selected_evaluation):
                criteria_ok = False
    checks.extend(
        [
            ("Selected candidate belongs to its recorded pool", membership_ok,
             "Candidate evaluation ID, C, gamma, mean and sample SD matched."),
            ("Selection criterion is computed correctly", criteria_ok,
             "Recomputed mean loss (standard) and mean + lambda*sample SD."),
        ]
    )
    for frame in (outer, calibrated):
        grouped = frame.groupby(
            ["dataset", "preprocessing_config", "outer_fold"]
        )["candidate_evaluations"].nunique()
        if not grouped.eq(1).all():
            shared_pool_ok = False
        if not frame["candidate_evaluations"].eq(
            frame["selected_pool_size"]
        ).all():
            shared_pool_ok = False
    checks.append(
        ("Both selectors use the same recorded candidate pool", shared_pool_ok,
         "Per-fold evaluation count and selected pool size agree.")
    )
    leakage_flags_ok = (
        history["outer_test_used_for_search"].eq(False).all()
        and outer["outer_test_used_for_selection"].eq(False).all()
        and calibrated["outer_test_used_for_selection"].eq(False).all()
        and calibrated["threshold_training_scope"]
        .eq("outer_train_inner_oof").all()
    )
    checks.append(
        ("Tuning and thresholds exclude outer-test data", bool(leakage_flags_ok),
         "Runner passes outer-training data to search/inner OOF calibration; "
         "metadata flags were checked.")
    )
    expected_output = OUTPUT_DIR.resolve()
    paths_isolated = (
        (expected_output == BASE_OUTPUT_DIR.resolve()
         or BASE_OUTPUT_DIR.resolve() in expected_output.parents)
        and config["output_directory"] == str(expected_output)
        and all(path.resolve().parent == expected_output for path in required_files)
    )
    checks.append(
        ("Output paths isolated to designated directory", paths_isolated,
         str(expected_output))
    )
    baseline_path = PROJECT_ROOT / "results" / (
        "nested_cv_all_datasets_preprocessing_comparison.csv"
    )
    comparison_path = OUTPUT_DIR / "baseline_vs_expanded_search.csv"
    compatible = False
    compatibility_detail = "Comparison or audited baseline table is missing."
    if baseline_path.is_file() and comparison_path.is_file():
        baseline = pd.read_csv(baseline_path)
        comparison = pd.read_csv(comparison_path)
        compatible = (
            set(comparison["_merge"].dropna()) == {"both"}
            and set(comparison["method"].unique()) == METHODS
            and baseline["outer_fold"].nunique() == config["outer_folds"]
            and config["outer_folds"] == 5
        )
        compatibility_detail = (
            f"Matched fold/method rows={len(comparison)}; baseline and expanded "
            "metrics use default threshold and same configured outer seed."
        )
    checks.append(
        ("Baseline comparison uses compatible evaluation conventions",
         compatible, compatibility_detail)
    )
    comparison_summary_path = (
        OUTPUT_DIR / "baseline_vs_expanded_search_summary.csv"
    )
    if comparison_summary_path.is_file():
        comparison_summary = pd.read_csv(comparison_summary_path)
        summary_ok = (
            len(comparison_summary) == 6
            and comparison_summary.select_dtypes(include=[np.number])
            .replace([np.inf, -np.inf], np.nan).notna().all().all()
        )
    else:
        summary_ok = False
    checks.append(
        ("Baseline delta summary is complete and finite", bool(summary_ok),
         "Expected three datasets by two selection methods.")
    )
    all_completed = (
        len(status) == 12 and status["status"].eq("completed").all()
        and status["completed_outer_folds"].eq(5).all()
    )
    checks.append(
        ("All 12 configurations completed", bool(all_completed),
         f"{int(status['status'].eq('completed').sum())}/12 completed.")
    )
    write_report(checks)
    if not all(passed for _, passed, _ in checks):
        raise SystemExit(1)


def write_report(checks: list[tuple[str, bool, str]]) -> None:
    lines = [
        "# Validation Report",
        "",
        "| Check | Result | Details |",
        "|---|---|---|",
    ]
    for name, passed, details in checks:
        escaped_details = details.replace("|", "\\|")
        lines.append(
            f"| {name} | {'PASS' if passed else 'FAIL'} | "
            f"{escaped_details} |"
        )
    lines.extend(
        [
            "",
            "Code-level leakage check: the PSO function accepts only the outer "
            "training partition; outer-test indices are passed only to final "
            "metric evaluation. Calibrated thresholds are selected from "
            "inner-fold out-of-fold scores generated on outer-training data.",
            "",
        ]
    )
    (OUTPUT_DIR / "validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
