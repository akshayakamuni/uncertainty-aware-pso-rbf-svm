"""Recompute descriptive summaries and figures from completed lambda runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAMBDA_DIR = REPO_ROOT / "experiments" / "uncertainty_aware_pso_lambda_multiseed"
MULTISEED_DIR = REPO_ROOT / "experiments" / "uncertainty_aware_pso_v2_multiseed"
DELIVERABLES_DIR = REPO_ROOT / "deliverables"
KEYS = ["seed", "dataset", "preprocessing_config", "outer_fold"]
RESULT_KEYS = KEYS + ["lambda"]
METRICS = {
    "outer_roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1_score": "F1",
    "absolute_inner_to_outer_loss_gap": "Absolute inner-to-outer loss gap",
}
PLOT_METRICS = ["outer_roc_auc", "balanced_accuracy", "sensitivity", "f1_score"]
PAIRED_SOURCE_COLUMNS = {
    "outer_roc_auc": "outer_roc_auc_lambda_minus_0",
    "balanced_accuracy": "balanced_accuracy_lambda_minus_0",
    "sensitivity": "sensitivity_lambda_minus_0",
    "specificity": "specificity_lambda_minus_0",
    "f1_score": "f1_score_lambda_minus_0",
    "inner_mean_loss": "inner_mean_loss_lambda_minus_0",
    "inner_std_loss": "inner_std_loss_lambda_minus_0",
    "inner_to_outer_loss_gap": "inner_to_outer_loss_gap_lambda_minus_0",
    "absolute_inner_to_outer_loss_gap":
        "absolute_inner_to_outer_loss_gap_lambda_minus_0",
}


def expected_keys(config: dict, seed: int) -> set[tuple]:
    return {
        (seed, dataset, preprocessing["name"], outer_fold, float(lambda_value))
        for dataset in config["datasets"]
        for preprocessing in config["preprocessing_configs"]
        for outer_fold in range(1, config["outer_folds"] + 1)
        for lambda_value in config["lambda_values"]
    }


def validate_result_domain(frame: pd.DataFrame, config: dict, name: str) -> None:
    missing_columns = set(RESULT_KEYS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{name} missing key columns: {sorted(missing_columns)}")
    if frame.duplicated(RESULT_KEYS).any():
        raise ValueError(f"{name} contains duplicate result keys.")

    actual = {
        (
            int(seed),
            dataset,
            preprocessing,
            int(outer_fold),
            float(lambda_value),
        )
        for seed, dataset, preprocessing, outer_fold, lambda_value
        in frame[RESULT_KEYS].itertuples(index=False, name=None)
    }
    configured_seeds = [int(value) for value in config["seeds"]]
    wanted = set().union(*(expected_keys(config, seed) for seed in configured_seeds))
    if actual != wanted:
        missing = len(wanted - actual)
        unexpected = len(actual - wanted)
        raise ValueError(
            f"{name} does not match the complete configured key domain "
            f"(missing={missing}, unexpected={unexpected})."
        )


def pair_against_lambda_zero(
    frame: pd.DataFrame, lambda_values: list[float]
) -> pd.DataFrame:
    """Create one matched nonzero-minus-zero row per saved outer-fold key."""
    metrics = list(METRICS) + [
        "inner_to_outer_loss_gap",
        "inner_mean_loss",
        "inner_std_loss",
    ]
    required = set(KEYS + ["lambda", *metrics])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot pair results; missing columns: {sorted(missing)}")
    if frame.duplicated(RESULT_KEYS).any():
        raise ValueError("Cannot pair results with duplicate keys.")

    index = KEYS
    wide = frame.pivot(index=index, columns="lambda", values=metrics)
    for value in lambda_values:
        if float(value) not in wide.columns.get_level_values(1):
            raise ValueError(f"Missing lambda={value} in paired source results.")
    if 0.0 not in wide.columns.get_level_values(1):
        raise ValueError("Lambda-zero baseline is missing.")

    rows = []
    for lambda_value in lambda_values:
        lambda_value = float(lambda_value)
        if lambda_value == 0.0:
            continue
        comparison = wide.xs(lambda_value, axis=1, level=1)
        baseline = wide.xs(0.0, axis=1, level=1)
        row = comparison.subtract(baseline)
        row.columns = [f"delta_{column}" for column in row.columns]
        row["lambda"] = lambda_value
        rows.append(row.reset_index())
    paired = pd.concat(rows, ignore_index=True)
    return paired.sort_values(RESULT_KEYS).reset_index(drop=True)


def validate_saved_pair_file(
    computed: pd.DataFrame, saved_path: Path, name: str
) -> None:
    saved = pd.read_csv(saved_path)
    if saved.duplicated(RESULT_KEYS).any():
        raise ValueError(f"{name} contains duplicate paired keys.")
    computed_keys = set(map(tuple, computed[RESULT_KEYS].itertuples(index=False, name=None)))
    saved_keys = set(map(tuple, saved[RESULT_KEYS].itertuples(index=False, name=None)))
    if computed_keys != saved_keys:
        raise ValueError(
            f"{name} paired key mismatch: "
            f"{len(computed_keys - saved_keys)} missing, "
            f"{len(saved_keys - computed_keys)} unexpected."
        )
    merged = computed.merge(
        saved, on=RESULT_KEYS, suffixes=("_computed", "_saved"), validate="one_to_one"
    )
    for metric, saved_column in PAIRED_SOURCE_COLUMNS.items():
        calculated_column = f"delta_{metric}"
        if calculated_column not in merged or saved_column not in merged:
            raise ValueError(
                f"{name} cannot verify {metric}; expected {saved_column}."
            )
        if not np.allclose(
            merged[calculated_column],
            merged[saved_column],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        ):
            raise ValueError(f"{name} disagrees with raw rows for {metric}.")


def seed_dataset_preprocessing_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Average folds within each seed/configuration before higher-level summaries."""
    grouping = ["seed", "dataset", "preprocessing_config", "lambda"]
    delta_columns = [column for column in paired if column.startswith("delta_")]
    return (
        paired.groupby(grouping, as_index=False)[delta_columns]
        .mean()
        .sort_values(grouping)
        .reset_index(drop=True)
    )


def seed_level_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Average matched folds/configurations inside each seed before averaging seeds."""
    delta_columns = [column for column in paired if column.startswith("delta_")]
    return (
        paired.groupby(["seed", "lambda"], as_index=False)[delta_columns]
        .mean()
        .sort_values(["lambda", "seed"])
        .reset_index(drop=True)
    )


def summarize_m1_m3(multiseed_dir: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    frames = []
    for seed in config["seeds"]:
        path = multiseed_dir / f"seed_{seed}" / "outer_fold_results.csv"
        frame = pd.read_csv(path)
        required = {
            "seed", "dataset", "preprocessing_config", "outer_fold", "method",
            "outer_roc_auc", "balanced_accuracy", "sensitivity", "specificity",
            "f1_score", "inner_to_outer_loss_gap",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS + ["method"]).any():
            raise ValueError(f"{path} contains duplicate M1/M3 fold rows.")
        expected = {
            (seed, dataset, preprocessing["name"], fold, method)
            for dataset in config["datasets"]
            for preprocessing in config["preprocessing_configs"]
            for fold in range(1, config["outer_folds"] + 1)
            for method in ("M1 Standard PSO", "M3 UA-PSO")
        }
        actual = set(
            map(tuple, frame[KEYS + ["method"]].itertuples(index=False, name=None))
        )
        if actual != expected:
            raise ValueError(f"{path} does not contain the complete M1/M3 fold domain.")
        frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    value_columns = [
        "outer_roc_auc", "balanced_accuracy", "sensitivity", "specificity",
        "f1_score", "inner_to_outer_loss_gap",
    ]
    wide = all_rows.pivot(
        index=KEYS,
        columns="method",
        values=value_columns,
    )
    deltas = pd.DataFrame(index=wide.index)
    for metric in value_columns:
        deltas[f"delta_{metric}"] = (
            wide[(metric, "M3 UA-PSO")] - wide[(metric, "M1 Standard PSO")]
        )
    gap_counts = {
        "smaller": int((deltas["delta_inner_to_outer_loss_gap"].abs() == 0).sum()),
        "m3_smaller_absolute": 0,
        "m3_larger_absolute": 0,
        "equal_absolute": 0,
    }
    m1_gap = wide[("inner_to_outer_loss_gap", "M1 Standard PSO")]
    m3_gap = wide[("inner_to_outer_loss_gap", "M3 UA-PSO")]
    gap_counts["m3_smaller_absolute"] = int((m3_gap.abs() < m1_gap.abs()).sum())
    gap_counts["m3_larger_absolute"] = int((m3_gap.abs() > m1_gap.abs()).sum())
    gap_counts["equal_absolute"] = int((m3_gap.abs() == m1_gap.abs()).sum())
    overall = deltas.mean().to_dict()
    return all_rows, {"overall_deltas": overall, "gap_counts": gap_counts}


def _seed_means(frame: pd.DataFrame, grouping: list[str], values: list[str]) -> pd.DataFrame:
    return frame.groupby(["seed", *grouping], as_index=False)[values].mean()


def create_figures(
    default: pd.DataFrame,
    calibrated: pd.DataFrame,
    paired: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed_means = _seed_means(paired, ["lambda"], [f"delta_{m}" for m in PLOT_METRICS])
    overall = seed_means.groupby("lambda", as_index=False)[
        [f"delta_{metric}" for metric in PLOT_METRICS]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, PLOT_METRICS):
        axis.plot(overall["lambda"], overall[f"delta_{metric}"], marker="o")
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set(title=METRICS[metric], xlabel="Uncertainty penalty (lambda)",
                 ylabel="Paired change from lambda=0")
        axis.set_xticks(sorted(paired["lambda"].unique()))
    fig.suptitle("Overall outer-fold metric changes relative to lambda=0")
    fig.savefig(figures_dir / "lambda_metric_changes.png", dpi=300)
    plt.close(fig)

    dataset_seed_means = _seed_means(
        paired, ["dataset", "lambda"], [f"delta_{m}" for m in PLOT_METRICS]
    )
    dataset_summary = dataset_seed_means.groupby(
        ["dataset", "lambda"], as_index=False
    )[[f"delta_{metric}" for metric in PLOT_METRICS]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, PLOT_METRICS):
        for dataset, group in dataset_summary.groupby("dataset"):
            axis.plot(group["lambda"], group[f"delta_{metric}"], marker="o", label=dataset)
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set(title=METRICS[metric], xlabel="Uncertainty penalty (lambda)",
                 ylabel="Paired change from lambda=0")
        axis.set_xticks(sorted(paired["lambda"].unique()))
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Dataset-level paired lambda comparisons")
    fig.savefig(figures_dir / "dataset_lambda_comparisons.png", dpi=300)
    plt.close(fig)

    gap_seed_means = _seed_means(
        paired, ["dataset", "lambda"], ["delta_absolute_inner_to_outer_loss_gap"]
    )
    gap_summary = gap_seed_means.groupby(
        ["dataset", "lambda"], as_index=False
    )["delta_absolute_inner_to_outer_loss_gap"].mean()
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for dataset, group in gap_summary.groupby("dataset"):
        axis.plot(
            group["lambda"],
            group["delta_absolute_inner_to_outer_loss_gap"],
            marker="o",
            label=dataset,
        )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set(
        title="Change in absolute inner-to-outer loss gap",
        xlabel="Uncertainty penalty (lambda)",
        ylabel="Paired change from lambda=0",
        xticks=sorted(paired["lambda"].unique()),
    )
    axis.legend(frameon=False)
    fig.savefig(figures_dir / "absolute_generalization_gap.png", dpi=300)
    plt.close(fig)

    metric_names = ["balanced_accuracy", "sensitivity", "specificity", "f1_score"]
    metric_frames = []
    for label, frame in (("Default threshold", default), ("Threshold calibrated", calibrated)):
        grouped = frame.groupby(["seed", "lambda"], as_index=False)[metric_names].mean()
        grouped["threshold_mode"] = label
        metric_frames.append(grouped)
    threshold_summary = pd.concat(metric_frames, ignore_index=True).groupby(
        ["lambda", "threshold_mode"], as_index=False
    )[metric_names].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, metric_names):
        for mode, group in threshold_summary.groupby("threshold_mode"):
            axis.plot(group["lambda"], group[metric], marker="o", label=mode)
        axis.set(title=METRICS[metric], xlabel="Uncertainty penalty (lambda)",
                 ylabel="Mean outer-fold metric", xticks=sorted(default["lambda"].unique()))
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Default-threshold and training-calibrated results")
    fig.savefig(figures_dir / "threshold_mode_comparison.png", dpi=300)
    plt.close(fig)


def format_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    display = frame[columns].copy()
    display.columns = [labels.get(column, column) for column in columns]
    rendered = []
    for _, row in display.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                cells.append("")
            elif column == "Lambda":
                cells.append(f"{float(value):g}")
            elif isinstance(value, (int, float, np.number)):
                cells.append(f"{float(value):+.4f}")
            else:
                cells.append(str(value))
        rendered.append(cells)
    header = list(display.columns)
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def build_findings_summary(
    default: pd.DataFrame,
    calibrated: pd.DataFrame,
    paired: pd.DataFrame,
    m1m3: pd.DataFrame,
    m1m3_summary: dict,
) -> str:
    seed_dataset = _seed_means(
        paired,
        ["dataset", "lambda"],
        [
            "delta_outer_roc_auc", "delta_balanced_accuracy", "delta_sensitivity",
            "delta_specificity", "delta_f1_score",
            "delta_absolute_inner_to_outer_loss_gap",
        ],
    )
    overall_seed = _seed_means(
        paired,
        ["lambda"],
        [
            "delta_outer_roc_auc", "delta_balanced_accuracy", "delta_sensitivity",
            "delta_specificity", "delta_f1_score",
            "delta_absolute_inner_to_outer_loss_gap",
        ],
    )
    overall = overall_seed.groupby("lambda", as_index=False).mean(numeric_only=True)
    display_columns = [
        "lambda", "delta_outer_roc_auc", "delta_balanced_accuracy",
        "delta_sensitivity", "delta_specificity", "delta_f1_score",
        "delta_absolute_inner_to_outer_loss_gap",
    ]
    labels = {
        "lambda": "Lambda",
        "delta_outer_roc_auc": "Δ ROC-AUC",
        "delta_balanced_accuracy": "Δ balanced accuracy",
        "delta_sensitivity": "Δ sensitivity",
        "delta_specificity": "Δ specificity",
        "delta_f1_score": "Δ F1",
        "delta_absolute_inner_to_outer_loss_gap": "Δ absolute gap",
    }
    dataset_table = seed_dataset.groupby(
        ["dataset", "lambda"], as_index=False
    )[display_columns[1:]].mean()
    threshold_metrics = [
        "balanced_accuracy", "sensitivity", "specificity", "f1_score"
    ]
    threshold_rows = []
    for mode, frame in (
        ("Default", default),
        ("Calibrated", calibrated),
    ):
        by_seed = frame.groupby(["seed", "lambda"], as_index=False)[
            threshold_metrics
        ].mean()
        by_lambda = by_seed.groupby("lambda", as_index=False)[
            threshold_metrics
        ].mean()
        by_lambda.insert(1, "threshold_mode", mode)
        threshold_rows.append(by_lambda)
    threshold_table = pd.concat(threshold_rows, ignore_index=True)
    gap = m1m3_summary["gap_counts"]
    deltas = m1m3_summary["overall_deltas"]
    lines = [
        "# Final saved-results analysis",
        "",
        "Generated from the saved final lambda fold-level CSVs and the completed "
        "three-seed M1/M3 fold outputs. The script does not fit models.",
        "",
        "## Protocol and aggregation",
        "",
        "Lambda 0 is M1's mean-inner-loss PSO objective; positive lambdas are "
        "M3 runs whose penalized fitness also guides personal/global-best updates. "
        "Each lambda comparison is paired by seed, dataset, preprocessing, and "
        "outer fold. Fold-level differences are retained, then averaged within "
        "seed before summarizing across the three seeds. Values below are "
        "descriptive; overlapping folds and reuse of the same datasets mean "
        "these rows are not independent samples.",
        "",
        "## Mean paired changes from lambda=0",
        "",
        "Default-threshold metrics and the signed/absolute gap use the complete "
        "matched lambda run data. Each table entry averages fold- and "
        "preprocessing-level differences within seed, then the three seed means.",
        "",
        format_table(overall, display_columns, labels),
        "",
        "## Dataset-level paired changes",
        "",
        format_table(
            dataset_table,
            ["dataset", *display_columns],
            {"dataset": "Dataset", **labels},
        ),
        "",
        "## Threshold modes",
        "",
        "Threshold-calibrated results use the saved outer-training inner-OOF "
        "thresholds and remain distinct from default-threshold results. "
        "Calibration changes classification metrics, not continuous-score "
        "ROC-AUC. The table averages dataset/configuration/fold rows within "
        "each seed, then reports the mean of the three seed means.",
        "",
        format_table(
            threshold_table,
            ["lambda", "threshold_mode", *threshold_metrics],
            {
                "lambda": "Lambda",
                "threshold_mode": "Threshold mode",
                "balanced_accuracy": "Balanced accuracy",
                "sensitivity": "Sensitivity",
                "specificity": "Specificity",
                "f1_score": "F1",
            },
        ),
        "",
        "## Three-seed M1 versus M3 at lambda=1",
        "",
        "Paired average M3-minus-M1 differences across matched saved rows: "
        + "; ".join(
            f"{metric.removeprefix('delta_')} {value:+.4f}"
            for metric, value in deltas.items()
            if metric in {
                "delta_outer_roc_auc", "delta_balanced_accuracy",
                "delta_sensitivity", "delta_specificity", "delta_f1_score",
            }
        )
        + ".",
        "",
        f"The absolute inner-to-outer gap was smaller for M3 in "
        f"{gap['m3_smaller_absolute']} of {gap['m3_smaller_absolute'] + gap['m3_larger_absolute'] + gap['equal_absolute']} "
        f"matched comparisons, larger in {gap['m3_larger_absolute']}, and equal "
        f"in {gap['equal_absolute']}. This is not consistent gap reduction.",
        "",
        "## Interpretation and limitations",
        "",
        "The saved results show changes that depend on dataset, preprocessing, "
        "lambda, and threshold mode; they do not establish that M3 is better "
        "than M1 or that a nonzero penalty reliably reduces overtuning. No "
        "p-values, confidence intervals, clinical-benefit claims, or claims of "
        "statistical significance are produced. The three seeds reuse the same "
        "datasets, and overlapping CV folds are not treated as independent.",
        "",
        "Historical v2 TruncatedSVD used `random_state: null`; these historical "
        "outputs were not regenerated. The final seeded lambda runs record the "
        "SVD random state per seed. The exact provenance/version and license "
        "terms for the local Pima CSV remain unresolved.",
        "",
        "## Figure files",
        "",
        "- `figures/lambda_metric_changes.png`",
        "- `figures/dataset_lambda_comparisons.png`",
        "- `figures/absolute_generalization_gap.png`",
        "- `figures/threshold_mode_comparison.png`",
        "",
        "The detailed paired rows recomputed from raw outputs are in "
        "`lambda_paired_differences_recomputed.csv`; seed-level summaries are "
        "in `lambda_seed_dataset_preprocessing_summary.csv`.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(repo_root: Path = REPO_ROOT, output_dir: Path | None = None) -> None:
    lambda_dir = repo_root / "experiments" / "uncertainty_aware_pso_lambda_multiseed"
    multiseed_dir = repo_root / "experiments" / "uncertainty_aware_pso_v2_multiseed"
    destination = output_dir or repo_root / "deliverables"
    config = json.loads((lambda_dir / "lambda_multiseed_config.json").read_text(encoding="utf-8"))
    default_frames, calibrated_frames = [], []
    for seed in config["seeds"]:
        seed_dir = lambda_dir / f"seed_{seed}"
        default_frames.append(pd.read_csv(seed_dir / "outer_fold_results.csv"))
        calibrated_frames.append(
            pd.read_csv(seed_dir / "threshold_calibrated_fold_results.csv")
        )
    default = pd.concat(default_frames, ignore_index=True)
    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    validate_result_domain(default, config, "Default-threshold fold results")
    validate_result_domain(calibrated, config, "Threshold-calibrated fold results")
    if set(map(tuple, default[RESULT_KEYS].itertuples(index=False, name=None))) != set(
        map(tuple, calibrated[RESULT_KEYS].itertuples(index=False, name=None))
    ):
        raise ValueError("Default and calibrated result keys do not match exactly.")

    lambda_values = [float(value) for value in config["lambda_values"]]
    default_paired = pair_against_lambda_zero(default, lambda_values)
    calibrated_paired = pair_against_lambda_zero(calibrated, lambda_values)
    validate_saved_pair_file(
        default_paired,
        lambda_dir / "paired_lambda_fold_differences.csv",
        "Saved default paired table",
    )
    validate_saved_pair_file(
        calibrated_paired,
        lambda_dir / "paired_lambda_calibrated_fold_differences.csv",
        "Saved calibrated paired table",
    )

    destination.mkdir(parents=True, exist_ok=True)
    paired_path = destination / "lambda_paired_differences_recomputed.csv"
    default_paired.to_csv(paired_path, index=False)
    calibrated_path = destination / "lambda_calibrated_paired_differences_recomputed.csv"
    calibrated_paired.to_csv(calibrated_path, index=False)
    seed_summary = seed_dataset_preprocessing_summary(default_paired)
    seed_summary.to_csv(
        destination / "lambda_seed_dataset_preprocessing_summary.csv", index=False
    )
    m1m3_rows, m1m3_summary = summarize_m1_m3(multiseed_dir, config)
    summary = build_findings_summary(
        default, calibrated, default_paired, m1m3_rows, m1m3_summary
    )
    (destination / "final_results_summary.md").write_text(summary, encoding="utf-8")
    create_figures(
        default,
        calibrated,
        default_paired,
        destination / "figures",
    )
    print(f"Validated {len(default)} default and {len(calibrated)} calibrated rows.")
    print(
        f"Recomputed {len(default_paired)} default and "
        f"{len(calibrated_paired)} calibrated paired comparisons."
    )
    print(f"Wrote analysis outputs to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_analysis(args.repo_root, args.output_dir)


if __name__ == "__main__":
    main()
