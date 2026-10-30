"""Independent checks for uncertainty-aware PSO-SVM v2 artifacts."""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
if str(OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUT_DIR))
import run_experiment as current_runner

DATASETS = {"diabetes", "breast_cancer", "heart_disease"}
CONFIGS = {"baseline_median", "mean_imputation", "pca_median", "svd_median"}
METHODS = {
    "M1 Standard PSO",
    "M2 Post-hoc UA",
    "M3 UA-PSO",
    "M4 Random",
}
LAMBDAS = {0.0, 0.25, 0.5, 1.0, 2.0}
METRICS = [
    "outer_roc_auc", "balanced_accuracy", "sensitivity",
    "specificity", "f1_score",
]
REQUIRED_FILES = [
    "experiment_config.json",
    "outer_fold_results.csv",
    "candidate_search_history.csv",
    "threshold_calibrated_fold_results.csv",
    "aggregated_results.csv",
    "paired_method_comparisons.csv",
    "lambda_sensitivity.csv",
    "runtime_summary.csv",
    "completion_log.csv",
    "failure_log.txt",
]


def pass_check(checks: list[tuple[str, bool, str]], name: str, ok: bool, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def ast_function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )


def stochastic_operations_are_seeded() -> bool:
    tree = ast.parse(inspect.getsource(current_runner))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    cv_calls = [
        node for node in calls
        if getattr(node.func, "id", getattr(node.func, "attr", "")) == "StratifiedKFold"
    ]
    rng_calls = [
        node for node in calls
        if getattr(node.func, "attr", "") == "default_rng"
    ]
    cv_is_seeded = len(cv_calls) == 2 and all(
        any(keyword.arg == "random_state" for keyword in call.keywords)
        and any(
            keyword.arg == "shuffle"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in cv_calls
    )
    rng_is_seeded = len(rng_calls) == 2 and all(
        len(call.args) == 1 for call in rng_calls
    )
    svd_model = current_runner.create_model(
        C=1.0,
        gamma=1.0,
        numeric_features=["feature"],
        categorical_features=[],
        config={
            "numeric_imputation": "median",
            "dimensionality_reduction": "svd",
        },
    )
    svd = svd_model.named_steps["preprocessing"].named_steps["reduction"]
    return (
        cv_is_seeded
        and rng_is_seeded
        and current_runner.SEED == current_runner.SVD_RANDOM_STATE
        and current_runner.SEED == 42
        and svd.random_state == current_runner.SEED
    )


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    missing = [name for name in REQUIRED_FILES if not (OUTPUT_DIR / name).is_file()]
    pass_check(
        checks, "Required results exist", not missing,
        "Missing: " + ", ".join(missing) if missing else "All required files present.",
    )
    if missing:
        write_report(checks)
        raise SystemExit(1)

    config = json.loads((OUTPUT_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    primary = pd.read_csv(OUTPUT_DIR / "outer_fold_results.csv")
    history = pd.read_csv(OUTPUT_DIR / "candidate_search_history.csv")
    calibrated = pd.read_csv(OUTPUT_DIR / "threshold_calibrated_fold_results.csv")
    lambdas = pd.read_csv(OUTPUT_DIR / "lambda_sensitivity.csv")
    runtime = pd.read_csv(OUTPUT_DIR / "runtime_summary.csv")
    status = pd.read_csv(OUTPUT_DIR / "completion_log.csv")

    expected_keys = {
        (dataset, config_name, fold, method)
        for dataset in DATASETS for config_name in CONFIGS
        for fold in range(1, 6) for method in METHODS
    }
    primary_keys = set(zip(
        primary["dataset"], primary["preprocessing_config"],
        primary["outer_fold"], primary["method"],
    ))
    calibrated_keys = set(zip(
        calibrated["dataset"], calibrated["preprocessing_config"],
        calibrated["outer_fold"], calibrated["method"],
    ))
    pass_check(
        checks, "All primary dataset/config/method/fold combinations represented",
        primary_keys == expected_keys and len(primary) == 240,
        f"Found {len(primary)} primary rows; expected 240.",
    )
    pass_check(
        checks, "All separate calibrated combinations represented",
        calibrated_keys == expected_keys and len(calibrated) == 240,
        f"Found {len(calibrated)} calibrated rows; expected 240.",
    )
    pass_check(
        checks, "No duplicate result keys",
        not primary.duplicated(
            ["dataset", "preprocessing_config", "outer_fold", "method"]
        ).any()
        and not calibrated.duplicated(
            ["dataset", "preprocessing_config", "outer_fold", "method"]
        ).any(),
        "Checked default and calibrated result keys.",
    )

    expected_search_labels = {
        "M1_standard_shared_pool",
        "M3_lambda_0.25",
        "M3_lambda_0.5",
        "M3_lambda_1",
        "M3_lambda_2",
        "M4_random_search",
    }
    search_key = [
        "dataset", "preprocessing_config", "outer_fold",
        "search_label", "evaluation",
    ]
    history_duplicates = history.duplicated(search_key).any()
    group_sizes = history.groupby(
        ["dataset", "preprocessing_config", "outer_fold", "search_label"]
    ).size()
    complete_searches = (
        len(history) == 3 * 4 * 5 * 6 * 48
        and set(history["search_label"].unique()) == expected_search_labels
        and group_sizes.eq(48).all()
    )
    pass_check(
        checks, "Candidate searches have documented 48-evaluation budgets",
        bool(complete_searches and not history_duplicates),
        f"Rows={len(history)}; expected 17,280; each of 360 searches has 48 rows.",
    )
    pass_check(
        checks, "No duplicate candidate evaluation keys",
        not history_duplicates,
        "Checked dataset/config/fold/search/evaluation identifiers.",
    )
    all_names = [
        ("primary", primary, METRICS + [
            "inner_mean_loss", "inner_std_loss", "selection_score",
            "inner_to_outer_loss_gap", "C", "gamma", "runtime_seconds",
        ]),
        ("calibrated", calibrated, METRICS + [
            "inner_mean_loss", "inner_std_loss", "selection_score",
            "inner_to_outer_loss_gap", "C", "gamma", "runtime_seconds",
        ]),
        ("lambda", lambdas, METRICS + [
            "inner_mean_loss", "inner_std_loss", "fitness", "C", "gamma",
            "search_runtime_seconds", "final_evaluation_runtime_seconds",
        ]),
    ]
    finite_ok = True
    ranges_ok = True
    for label, frame, columns in all_names:
        if not set(columns).issubset(frame.columns) or frame.empty:
            finite_ok = ranges_ok = False
            continue
        finite_ok &= bool(np.isfinite(frame[columns].to_numpy(dtype=float)).all())
        ranges_ok &= all(frame[column].between(0.0, 1.0).all() for column in METRICS)
        ranges_ok &= frame["inner_mean_loss"].between(0.0, 1.0).all()
        ranges_ok &= frame["inner_std_loss"].ge(0.0).all()
        ranges_ok &= frame["C"].between(1e-3, 1e3).all()
        ranges_ok &= frame["gamma"].between(1e-3, 1e3).all()
        ranges_ok &=         frame[
            "runtime_seconds" if label != "lambda" else
            "search_runtime_seconds"
        ].ge(0.0).all()
    pass_check(checks, "All result metrics and runtimes finite", finite_ok,
               "Checked default, calibrated, and lambda sensitivity rows.")
    pass_check(checks, "Metrics and hyperparameters within valid ranges", ranges_ok,
               "AUC/class metrics/losses in [0,1], SD/runtime nonnegative, C/gamma in bounds.")

    history_lookup = {
        (row.dataset, row.preprocessing_config, int(row.outer_fold),
         row.search_label, int(row.evaluation)): row
        for row in history.itertuples(index=False)
    }
    membership_ok = True
    criterion_ok = True
    for frame in (primary, calibrated):
        for row in frame.itertuples(index=False):
            key = (
                row.dataset, row.preprocessing_config, int(row.outer_fold),
                row.candidate_pool_id, int(row.selected_evaluation),
            )
            candidate = history_lookup.get(key)
            if candidate is None:
                membership_ok = False
                continue
            for name in ("C", "gamma", "inner_mean_loss", "inner_std_loss"):
                history_name = {
                    "inner_mean_loss": "mean_inner_loss",
                    "inner_std_loss": "std_inner_loss",
                }.get(name, name)
                if not np.isclose(getattr(row, name), getattr(candidate, history_name)):
                    membership_ok = False
            expected_lambda = (
                1.0 if row.method in {"M2 Post-hoc UA", "M3 UA-PSO"} else 0.0
            )
            expected_score = row.inner_mean_loss + expected_lambda * row.inner_std_loss
            if not np.isclose(row.selection_score, expected_score, atol=1e-12):
                criterion_ok = False
            pool = history[
                (history["dataset"] == row.dataset)
                & (history["preprocessing_config"] == row.preprocessing_config)
                & (history["outer_fold"] == row.outer_fold)
                & (history["search_label"] == row.candidate_pool_id)
            ]
            criterion_values = (
                pool["mean_inner_loss"].to_numpy()
                + expected_lambda * pool["std_inner_loss"].to_numpy()
            )
            expected_eval = int(pool.iloc[int(np.argmin(criterion_values))]["evaluation"])
            if expected_eval != int(row.selected_evaluation):
                criterion_ok = False
    pass_check(checks, "Selected candidates belong to their recorded pools",
               membership_ok, "Evaluation IDs and candidate parameters/loss statistics matched.")
    pass_check(checks, "Final selection criteria recompute correctly",
               criterion_ok, "Mean loss for M1/M4; mean + lambda*sample SD for M2/M3.")

    shared_pool = True
    primary_lookup = primary.set_index(
        ["dataset", "preprocessing_config", "outer_fold", "method"]
    )
    for dataset, config_name, fold in {
        (dataset, config_name, fold)
        for dataset, config_name, fold, _method in expected_keys
    }:
        m1 = primary_lookup.loc[(dataset, config_name, fold, "M1 Standard PSO")]
        m2 = primary_lookup.loc[(dataset, config_name, fold, "M2 Post-hoc UA")]
        shared_pool &= (
            m1["candidate_pool_id"] == m2["candidate_pool_id"]
            and m1["candidate_pool_evaluations"] == m2["candidate_pool_evaluations"] == 48
        )
    pass_check(checks, "M1 and M2 use an identical candidate pool",
               shared_pool, "Same M1 history ID and all 48 evaluations per matched fold.")

    sample_sd_ok = True
    for _, row in history.iterrows():
        fold_losses = np.asarray(json.loads(row["fold_losses"]), dtype=float)
        if (
            fold_losses.size != config["inner_folds"]
            or not np.isfinite(fold_losses).all()
            or not np.isclose(np.mean(fold_losses), row["mean_inner_loss"], atol=1e-12)
            or not np.isclose(np.std(fold_losses, ddof=1), row["std_inner_loss"], atol=1e-12)
        ):
            sample_sd_ok = False
            break
        expected_fitness = row["mean_inner_loss"] + row["lambda"] * row["std_inner_loss"]
        if not np.isclose(expected_fitness, row["fitness"], atol=1e-12):
            sample_sd_ok = False
            break
    pass_check(checks, "Fitness uses sample SD and recorded fold losses",
               sample_sd_ok, "Recomputed means, ddof=1 SDs, and lambda-penalized fitness.")

    trajectory_ok = True
    pso = history[history["method_family"] == "PSO"]
    for _, group in pso.groupby(
        ["dataset", "preprocessing_config", "outer_fold", "search_label"],
        sort=False,
    ):
        group = group.sort_values("evaluation")
        global_best = math.inf
        global_eval = 0
        particle_state: dict[int, tuple[float, int]] = {}
        for _, row in group.iterrows():
            fold_losses = np.asarray(json.loads(row["fold_losses"]), dtype=float)
            current_fitness = float(
                np.mean(fold_losses)
                + row["lambda"] * np.std(fold_losses, ddof=1)
            )
            previous_fitness, previous_eval = particle_state.get(
                int(row["particle"]), (math.inf, 0)
            )
            expected_particle_fitness, expected_particle_eval = (
                current_fitness, int(row["evaluation"])
            ) if current_fitness < previous_fitness else (previous_fitness, previous_eval)
            if (
                expected_particle_eval != int(row["personal_best_evaluation"])
                or not np.isclose(
                    expected_particle_fitness, row["personal_best_fitness"],
                    rtol=0.0, atol=1e-15,
                )
            ):
                trajectory_ok = False
                break
            particle_state[int(row["particle"])] = (
                expected_particle_fitness, expected_particle_eval
            )
            if current_fitness < global_best:
                global_best, global_eval = current_fitness, int(row["evaluation"])
            if (
                global_eval != int(row["global_best_evaluation"])
                or not np.isclose(
                    global_best, row["global_best_fitness"],
                    rtol=0.0, atol=1e-15,
                )
            ):
                trajectory_ok = False
                break
        if not trajectory_ok:
            break
    script_tree = ast.parse(
        (OUTPUT_DIR / "run_experiment.py").read_text(encoding="utf-8")
    )
    pso_function = ast_function(script_tree, "run_pso_search")
    update_calls = (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "update_best"
            for node in ast.walk(pso_function)
        )
        if pso_function else 0
    )
    m3_update_ok = trajectory_ok and update_calls >= 2
    pass_check(checks, "M3 personal/global bests use penalized fitness",
               m3_update_ok,
               "Replayed logged best-so-far traces and verified two update_best calls in PSO loop.")

    lambda_keyset = set(zip(
        lambdas["dataset"], lambdas["preprocessing_config"],
        lambdas["outer_fold"], lambdas["lambda"],
    ))
    expected_lambda_keys = {
        (dataset, config_name, fold, lam)
        for dataset in DATASETS for config_name in CONFIGS
        for fold in range(1, 6) for lam in LAMBDAS
    }
    lambda_ok = (
        lambda_keyset == expected_lambda_keys
        and len(lambdas) == 300
        and lambdas["candidate_evaluations"].eq(48).all()
    )
    for _, row in lambdas.iterrows():
        pool_label = (
            "M1_standard_shared_pool" if row["lambda"] == 0.0
            else f"M3_lambda_{row['lambda']:g}"
        )
        pool = history[
            (history["dataset"] == row["dataset"])
            & (history["preprocessing_config"] == row["preprocessing_config"])
            & (history["outer_fold"] == row["outer_fold"])
            & (history["search_label"] == pool_label)
        ]
        if len(pool) != 48:
            lambda_ok = False
            continue
        scores = []
        for _, candidate in pool.iterrows():
            losses = np.asarray(json.loads(candidate["fold_losses"]), dtype=float)
            scores.append(
                float(np.mean(losses) + row["lambda"] * np.std(losses, ddof=1))
            )
        if int(pool.iloc[int(np.argmin(scores))]["evaluation"]) != int(row["selected_evaluation"]):
            lambda_ok = False
    pass_check(checks, "M3 lambda sensitivity reruns the penalized search",
               bool(lambda_ok), "All 5 declared lambdas have 60 fold/config rows and 48-evaluation pools.")

    initial_positions_ok = True
    initial = history[
        (history["method_family"] == "PSO") & (history["iteration"] == 1)
    ]
    for _, group in initial.groupby(
        ["dataset", "preprocessing_config", "outer_fold", "particle"]
    ):
        if (
            not np.isclose(group["initial_log10_C"].max(), group["initial_log10_C"].min())
            or not np.isclose(group["initial_log10_gamma"].max(), group["initial_log10_gamma"].min())
        ):
            initial_positions_ok = False
    pass_check(checks, "Matched initial PSO positions across objectives",
               initial_positions_ok, "Matched seed generated equal particle start coordinates per fold.")
    pass_check(
        checks, "Current runner controls stochastic operation seeds",
        stochastic_operations_are_seeded() and config.get("seed") == current_runner.SEED,
        "Current source seeds outer/inner stratified splits, both search RNGs, "
        "and TruncatedSVD with seed 42. Historical config retains "
        f"svd_random_state={config.get('svd_random_state')!r}; stored outputs "
        "remain from that original run and were not regenerated.",
    )

    random_pool = history[history["method_family"] == "Random"]
    random_ok = (
        len(random_pool) == 3 * 4 * 5 * 48
        and random_pool["search_label"].eq("M4_random_search").all()
        and random_pool["personal_best_evaluation"].isna().all()
        and random_pool["global_best_evaluation"].isna().all()
    )
    pass_check(checks, "M4 is a separate equal-budget random-search control",
               random_ok, "Random pool has 48 in-bound candidates per fold and no PSO-best updates.")

    leakage_flags = (
        history["outer_test_used_for_search"].eq(False).all()
        and primary["outer_test_used_for_selection"].eq(False).all()
        and calibrated["outer_test_used_for_selection"].eq(False).all()
        and calibrated["threshold_training_scope"].eq("outer_train_inner_oof").all()
    )
    threshold_function = ast_function(script_tree, "score_candidate")
    cv_call_ok = False
    threshold_call_ok = False
    if threshold_function:
        for node in ast.walk(threshold_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "cross_val_predict":
                    cv_call_ok |= (
                        len(node.args) >= 3
                        and isinstance(node.args[1], ast.Name)
                        and node.args[1].id == "X_train"
                        and isinstance(node.args[2], ast.Name)
                        and node.args[2].id == "y_train"
                    )
                if node.func.id == "choose_threshold":
                    threshold_call_ok |= (
                        len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "y_train"
                    )
    leakage_ok = leakage_flags and cv_call_ok and threshold_call_ok
    pass_check(checks, "Tuning and thresholds exclude outer-test data",
               bool(leakage_ok),
               "Checked runner calls use only X_train/y_train for search and inner OOF thresholds.")
    pass_check(checks, "Default-threshold and calibrated outputs remain separate",
               primary["threshold_training_scope"].eq("svc_default").all()
               and calibrated["threshold_training_scope"].eq("outer_train_inner_oof").all(),
               "Separate result files and explicit threshold-scope fields.")

    runtime_ok = (
        len(runtime) == 3 * 4 * 5 * 6
        and runtime["candidate_evaluations"].eq(48).all()
        and np.isfinite(runtime["runtime_seconds"]).all()
    )
    pass_check(checks, "Search runtimes and evaluation counts recorded",
               runtime_ok, f"Runtime rows={len(runtime)}; expected 360 searches.")
    all_completed = (
        len(status) == 12
        and status["status"].eq("completed").all()
        and status["completed_outer_folds"].eq(5).all()
    )
    pass_check(checks, "All dataset/configuration runs completed",
               all_completed, f"{int(status['status'].eq('completed').sum())}/12 completed.")
    output_isolated = config["output_directory"] == str(OUTPUT_DIR.resolve())
    pass_check(checks, "Output directory isolated from prior experiment folders",
               output_isolated, str(OUTPUT_DIR.resolve()))
    temp_files = list(OUTPUT_DIR.glob(".*.tmp"))
    pass_check(checks, "No incomplete temporary CSV files remain",
               not temp_files, ", ".join(item.name for item in temp_files) or "None.")

    write_report(checks)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def write_report(checks: list[tuple[str, bool, str]]) -> None:
    lines = [
        "# v2 Validation Report",
        "",
        "| Check | Result | Details |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        escaped_detail = detail.replace("|", "\\|")
        lines.append(
            f"| {name} | {'PASS' if ok else 'FAIL'} | "
            f"{escaped_detail} |"
        )
    lines.extend(
        [
            "",
            "The best-update validation independently replays the persisted "
            "per-particle and global best fitness traces and checks the runner "
            "calls the shared update function in its PSO loop. The companion "
            "`test_experiment.py` provides a deterministic synthetic test in "
            "which the mean-loss and penalized-fitness choices differ.",
            "",
            "The SVD seed check validates the current runner only. The recorded "
            "experiment config and result files retain their original "
            "`svd_random_state: null`; these historical results were not "
            "regenerated or independently reproduced.",
            "",
        ]
    )
    (OUTPUT_DIR / "validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
