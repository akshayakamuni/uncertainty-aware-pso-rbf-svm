"""Post-hoc lambda sensitivity using saved PSO candidate histories.

This script does NOT rerun PSO. It re-selects candidates from the saved unique
candidate pool for each lambda, then refits/evaluates those candidates on the
same outer folds. This isolates the effect of the final candidate-selection
penalty conditional on the original PSO search histories.

Run from repository root:
    python -m src.evaluation.posthoc_lambda_sensitivity
"""
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score
)
from src.data_loader import get_dataset
from src.evaluation.nested_cv_experiment import (
    RANDOM_STATE, OUTER_FOLDS, PREPROCESSING_CONFIGS, RESULTS_DIR,
    create_model,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = RESULTS_DIR / "additional_analysis"
OUT.mkdir(parents=True, exist_ok=True)
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
DATASETS = ["diabetes", "breast_cancer", "heart_disease"]
METRICS = ["outer_roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1_score", "overtuning_gap"]


def unique_candidates(history):
    # Match the original implementation's duplicate removal and first-seen rule.
    seen, candidates = set(), []
    for _, row in history.iterrows():
        key = (round(float(row.C), 10), round(float(row.gamma), 10))
        if key not in seen:
            seen.add(key)
            candidates.append({
                "C": float(row.C), "gamma": float(row.gamma),
                "mean_inner_loss": float(row.mean_inner_loss),
                "std_inner_loss": float(row.std_inner_loss),
            })
    return candidates


def metrics_for(model, X_train, y_train, X_test, y_test, candidate, config):
    model = create_model(
        C=candidate["C"], gamma=candidate["gamma"],
        numeric_features=config["numeric_features"],
        categorical_features=config["categorical_features"],
        numeric_imputation=config["numeric_imputation"],
        dimensionality_reduction=config["dimensionality_reduction"],
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    y_test = np.asarray(y_test, dtype=int)
    auc = float(roc_auc_score(y_test, scores))
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if tp + fn else 0.0
    spec = float(tn / (tn + fp)) if tn + fp else 0.0
    outer_loss = 1.0 - auc
    return {
        "outer_roc_auc": auc,
        "outer_loss": outer_loss,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "sensitivity": sens,
        "specificity": spec,
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "overtuning_gap": outer_loss - candidate["mean_inner_loss"],
        "C": candidate["C"], "gamma": candidate["gamma"],
        "inner_mean_loss": candidate["mean_inner_loss"],
        "inner_std_loss": candidate["std_inner_loss"],
        "selection_score": candidate["mean_inner_loss"] + candidate["lambda"] * candidate["std_inner_loss"],
    }


def main():
    all_rows = []
    for dataset in DATASETS:
        X, y, categorical = get_dataset(dataset)
        numeric = [c for c in X.columns if c not in categorical]
        outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_indices = list(outer.split(X, y))
        for cfg in PREPROCESSING_CONFIGS:
            hist_path = RESULTS_DIR / f"nested_cv_{dataset}_{cfg['name']}_search_history.csv"
            if not hist_path.exists():
                raise FileNotFoundError(f"Required saved search history missing: {hist_path}")
            history = pd.read_csv(hist_path)
            for fold, (train_idx, test_idx) in enumerate(split_indices, start=1):
                fold_hist = history.loc[history.outer_fold == fold]
                candidates = unique_candidates(fold_hist)
                if not candidates:
                    raise ValueError(f"No candidates in {hist_path} for outer fold {fold}")
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model_cfg = {
                    "numeric_features": numeric,
                    "categorical_features": categorical,
                    "numeric_imputation": cfg["numeric_imputation"],
                    "dimensionality_reduction": cfg["dimensionality_reduction"],
                }
                # Standard selection is the lambda=0 selection. Compute once per fold.
                for lam in LAMBDAS:
                    chosen = min(candidates, key=lambda c: c["mean_inner_loss"] + lam*c["std_inner_loss"])
                    candidate = {**chosen, "lambda": lam}
                    met = metrics_for(None, X_train, y_train, X_test, y_test, candidate, model_cfg)
                    all_rows.append({
                        "dataset": dataset, "preprocessing_config": cfg["name"],
                        "outer_fold": fold, "lambda": lam,
                        "method": "Standard" if lam == 0.0 else "Uncertainty-aware",
                        "candidate_pool_size": len(candidates), **met,
                    })
                print(f"Finished {dataset} | {cfg['name']} | fold {fold}")

    folds = pd.DataFrame(all_rows)
    folds_path = OUT / "lambda_sensitivity_outer_folds.csv"
    folds.to_csv(folds_path, index=False)

    summary = folds.groupby(["dataset", "preprocessing_config", "lambda"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = OUT / "lambda_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Paired fold deltas versus lambda=0 for each dataset/configuration.
    base = folds.loc[folds["lambda"] == 0].drop(columns=["method", "candidate_pool_size", "lambda"])
    base = base.rename(columns={m: f"{m}_standard" for m in METRICS})
    deltas = folds.loc[folds["lambda"] > 0].merge(
        base[["dataset", "preprocessing_config", "outer_fold"] + [f"{m}_standard" for m in METRICS]],
        on=["dataset", "preprocessing_config", "outer_fold"], how="left"
    )
    for m in METRICS:
        deltas[f"delta_{m}"] = deltas[m] - deltas[f"{m}_standard"]
    delta_summary = deltas.groupby(["dataset", "preprocessing_config", "lambda"])[[f"delta_{m}" for m in METRICS]].agg(["mean", "std"])
    delta_summary.columns = [f"{metric}_{stat}" for metric, stat in delta_summary.columns]
    delta_summary = delta_summary.reset_index()
    delta_path = OUT / "lambda_sensitivity_paired_deltas.csv"
    delta_summary.to_csv(delta_path, index=False)

    # Compare recomputed lambda=1 results against saved primary experiment metrics.
    original_path = RESULTS_DIR / "nested_cv_all_datasets_preprocessing_comparison.csv"
    original = pd.read_csv(original_path)
    orig = original.loc[original.method == "Uncertainty-aware", ["dataset", "preprocessing_config", "outer_fold"] + METRICS]
    recomputed = folds.loc[folds["lambda"] == 1.0, ["dataset", "preprocessing_config", "outer_fold"] + METRICS]
    chk = orig.merge(recomputed, on=["dataset", "preprocessing_config", "outer_fold"], suffixes=("_saved", "_recomputed"))
    for m in METRICS:
        chk[f"absdiff_{m}"] = (chk[f"{m}_saved"] - chk[f"{m}_recomputed"]).abs()
    chk.to_csv(OUT / "lambda1_reproduction_check.csv", index=False)
    check_cols = [c for c in chk.columns if c.startswith("absdiff_")]
    max_diff = float(chk[check_cols].max().max()) if len(chk) else float("nan")

    print("\nSaved:", folds_path)
    print("Saved:", summary_path)
    print("Saved:", delta_path)
    print("Saved: lambda1_reproduction_check.csv")
    print("Max absolute difference for lambda=1 reproduction:", max_diff)

if __name__ == "__main__":
    main()
