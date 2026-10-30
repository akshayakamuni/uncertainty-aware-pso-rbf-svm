# Expanded Search Findings

This is a descriptive comparison. PSO minimizes mean inner-fold loss; uncertainty-aware selection is applied afterward to the same recorded candidate pool.

## Default-threshold results by dataset

Each mean averages the four preprocessing configurations and five outer folds for that dataset. Per-configuration fold means and standard deviations are in `aggregated_results.csv`.

| Dataset | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap | Selector runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer | Standard | 0.9945 | 0.9653 | 0.9447 | 0.9860 | 0.9595 | 0.0018 | 0.02 |
| breast_cancer | Uncertainty-aware | 0.9950 | 0.9665 | 0.9470 | 0.9860 | 0.9610 | 0.0010 | 0.02 |
| diabetes | Standard | 0.8333 | 0.6972 | 0.5144 | 0.8800 | 0.5777 | 0.0010 | 0.03 |
| diabetes | Uncertainty-aware | 0.8342 | 0.6921 | 0.4982 | 0.8860 | 0.5613 | -0.0016 | 0.03 |
| heart_disease | Standard | 0.9046 | 0.8153 | 0.7354 | 0.8952 | 0.7707 | -0.0015 | 0.04 |
| heart_disease | Uncertainty-aware | 0.9050 | 0.8113 | 0.7319 | 0.8906 | 0.7665 | -0.0029 | 0.04 |

## Matched baseline deltas (expanded minus existing baseline)

Default-threshold comparison matched on dataset, configuration, outer fold, and selector. Means below cover four configurations by five folds and are descriptive.

| Dataset | Method | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ loss gap |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | Standard | 0.0021 | 0.0832 | 0.1720 | -0.0056 | 0.1804 | -0.0005 |
| breast_cancer | Uncertainty-aware | 0.0026 | 0.0854 | 0.1743 | -0.0035 | 0.1836 | -0.0012 |
| diabetes | Standard | 0.0027 | 0.0339 | 0.0778 | -0.0100 | 0.0944 | -0.0005 |
| diabetes | Uncertainty-aware | -0.0009 | 0.0370 | 0.0876 | -0.0135 | 0.1053 | 0.0042 |
| heart_disease | Standard | 0.0098 | 0.0394 | 0.1107 | -0.0318 | 0.1107 | -0.0082 |
| heart_disease | Uncertainty-aware | 0.0103 | 0.0194 | 0.0448 | -0.0061 | 0.0497 | -0.0079 |

## Completion and limitations

- Successful configurations: 12 of 12.
- Failed configurations: 0.
- Total wall-clock runtime was approximately 455 seconds (7.6 minutes); per-configuration and per-fold elapsed times are recorded in `completion_log.csv` and `experiment.log`. Fold elapsed time includes search and both selector evaluations; per-row `runtime_seconds` is the selected-model evaluation stage.
- The first full-run attempt stopped before completing any fold due to a runner NameError; its logs are preserved under `../failed_attempt_1/`. The corrected run is this complete `run_2`.
- These aggregate means pool different datasets and preprocessing configurations and are descriptive, not a pooled clinical estimate. Within each dataset, fold/configuration observations also overlap and must not be treated as independent samples.
- The five outer folds overlap in training data and are not independent samples; no statistical significance is claimed.
- The default-threshold baseline comparison is convention-matched; calibrated outputs are reported separately.
