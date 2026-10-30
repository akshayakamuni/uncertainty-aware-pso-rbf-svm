# Expanded PSO Search Experiment (v1)

This experiment tests a moderately larger PSO search for RBF-SVMs while
preserving the existing nested-CV protocol and preprocessing. It compares
standard selection with uncertainty-aware selection from the **same recorded
PSO candidate pool**. PSO particle movement and personal/global best updates
minimize mean inner-fold ROC-AUC loss only; the uncertainty penalty is applied
after search when choosing a candidate.

## Configuration

- Datasets: `diabetes`, `breast_cancer`, and `heart_disease`, loaded with the
  existing target mappings, missing-value handling, and feature definitions.
- Preprocessing: `baseline_median`, `mean_imputation`, `pca_median`, and
  `svd_median`, using the existing leakage-safe pipeline. PCA retains 95% of
  variance; SVD uses five components.
- Nested stratified CV: five outer folds and five inner folds, shuffled using
  the established seed derivation from seed 42.
- PSO: eight particles × six iterations, bounds `[-3, 3]` for both
  `log10(C)` and `log10(gamma)`, inertia 0.7, cognitive/social coefficients
  1.5. A complete search records 48 particle evaluations per outer-fold
  search; duplicate parameter points remain recorded.
- Objective: mean of the five inner-fold losses, each `1 - ROC-AUC` from
  continuous SVM decision scores. Candidate variability is sample standard
  deviation (`ddof=1`). Primary uncertainty penalty is lambda = 1.
- Primary classification metrics use SVC's default decision threshold.
  Separate threshold-calibrated results select thresholds by inner out-of-fold
  balanced accuracy using only outer-training data. These calibrated outputs
  are not mixed with the default-threshold baseline comparison.
- Inner-to-outer loss gap is `(1 - outer ROC-AUC) - selected mean inner loss`.

The outer split seed and fold-specific inner/PSO seed derivation match the
existing experiment. Comparisons are descriptive: the five overlapping outer
folds are not independent samples and no significance claims are made.

## Run

From the project directory, with the repository requirements installed:

```powershell
$env:EXPANDED_SEARCH_OUTPUT_DIR = 'experiments\expanded_search_v1\run_new'
python experiments\expanded_search_v1\run_expanded_search.py
python experiments\expanded_search_v1\validate_outputs.py
```

The runner checkpoints after each completed dataset/configuration/fold. A
failure is logged and later configurations continue where possible; rerunning
starts a fresh run and is not an automatic resume. Do not run it if the
designated output directory already contains results you need to preserve.
Choose a new, unused child directory for a fresh run; the runner and validator
enforce that the directory remains inside this experiment folder. To validate
the completed run from this workflow, use:

```powershell
$env:EXPANDED_SEARCH_OUTPUT_DIR = 'experiments\expanded_search_v1\run_2'
python experiments\expanded_search_v1\validate_outputs.py
```

## Outputs

- `outer_fold_results.csv`: default-threshold fold results for both selectors.
- `threshold_calibrated_fold_results.csv`: separately calibrated fold results.
- `candidate_search_history.csv`: all evaluated candidates and five inner-fold
  losses per candidate.
- `aggregated_results.csv`: means and fold standard deviations.
- `selection_paired_differences.csv`: fold-matched uncertainty-aware minus
  standard metric differences.
- `baseline_vs_expanded_search.csv`: fold-matched comparison with the existing
  default-threshold nested-CV results, when the baseline table is available.
- `baseline_vs_expanded_search_summary.csv`: descriptive matched metric deltas
  by dataset and selection method.
- `completion_log.csv`, `experiment.log`, and `failure_log.txt`: status,
  elapsed time, runtime, and failures.
- `runtime_seconds` in fold results measures the selected model's evaluation
  stage. Full search + both selectors' per-fold elapsed times are logged in
  `experiment.log`; configuration totals are in `completion_log.csv`.
- `validation_report.md`: checks on coverage, duplicate keys, metric ranges,
  selection provenance, and baseline compatibility.
- `findings_summary.md`: generated descriptive findings and limitations.

Run-specific results and logs are stored in the selected child directory.
The source modules, datasets, and existing baseline outputs are read only.
