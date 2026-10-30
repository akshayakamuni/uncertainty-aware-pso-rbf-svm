# Uncertainty-Aware PSO-SVM v2

Independent experiment. Existing source, results, datasets, and expanded-search
outputs are read only. The runner refuses to start if its result files already
exist in this directory.

## Question and method distinction

For each candidate, the inner-fold losses are `1 - ROC-AUC` from continuous
SVC decision scores. The sample standard deviation uses `ddof=1` across five
folds. The primary v2 method optimizes
`J_lambda = mean_loss + lambda * sample_sd` during particle evaluation and
personal/global-best updates, then selects the minimum-J candidate. It is not
the existing post-hoc selector.

Four methods use the same outer/inner splits and 48-evaluation budget:

1. **M1 Standard PSO-SVM:** PSO fitness and final selection use mean loss.
2. **M2 Post-hoc uncertainty-aware:** the identical M1 history is selected
   using `mean + 1 * sample_sd`; it does not change the M1 swarm trajectory.
3. **M3 Uncertainty-aware PSO-SVM:** PSO fitness, personal/global bests, and
   final selection use penalized fitness, with lambda 1 primary.
4. **M4 Random-search control:** 48 uniformly sampled points within the same
   log10 bounds, selected by mean loss; it is not PSO.

M1/M2 share the exact candidate pool. Each M3 lambda run and M1 start from
matched particle positions and same-seed random-number streams, although paths
can diverge as soon as objective rankings differ. Lambda sensitivity reruns
M3 at lambda 0.25, 1, and 2, reuses M1 at lambda 0, and separately runs lambda
0.5. No lambda is chosen from outer-test performance.

## Protocol

- Datasets and mappings: existing loader (`breast_cancer`, `diabetes`,
  `heart_disease`).
- Preprocessing: existing pipeline, fitted inside each CV training fold:
  median, mean, PCA retaining 95% variance, and median + five-component SVD.
- Outer/inner CV: stratified shuffled 5-fold. Outer seed 42; inner split and
  optimizer seed are 42 + outer-fold number, consistently with the baseline.
- Search bounds: `log10(C), log10(gamma) in [-3, 3]`; 8 particles × 6
  iterations; inertia 0.7; cognitive/social terms 1.5.
- Fitness sample SD convention: sample SD (`ddof=1`); each evaluation requires
  all five finite fold losses. A failed fold raises and is logged; it is never
  replaced by a success-shaped value.
- The default-threshold evaluation uses SVC predictions. Separately saved
  calibrated metrics choose threshold by inner-training out-of-fold balanced
  accuracy, using the existing tie-break rule (closest to zero). The known
  inner-OOF/full-training refit score-scale mismatch remains a limitation.
- Seeds 42 and fold-offset seeds are used for split generation and search.
  The existing read-only preprocessing implementation constructs
  `TruncatedSVD` with `random_state=None`; therefore SVD solver randomness is
  not controlled in the completed historical run. The v2 runner now sets the
  SVD estimator's `random_state` from the configured seed (42) for future
  runs. The existing outputs were not regenerated or independently
  reproduced. The default PCA solver selected for these datasets during the
  audit was deterministic `covariance_eigh`; the default SVC decision/predict
  path also has no stochastic probability calibration enabled.
- Loss gap: `(1 - outer ROC-AUC) - selected mean inner-fold loss`.

The one-seed initial experiment does not establish seed stability. Repeated
seed study is deliberately not included in this run because the full lambda
grid multiplies the search workload; selected hyperparameter variation across
folds is descriptive only and does not answer H3.

## Run and validate

From the project directory (PowerShell):

```powershell
python experiments\uncertainty_aware_pso_v2\test_experiment.py
python experiments\uncertainty_aware_pso_v2\run_experiment.py
python experiments\uncertainty_aware_pso_v2\validate_outputs.py
```

The runner checkpoints completed candidate evaluations and folds and records
errors. If interrupted or a result file exists, preserve this directory and
copy the scripts to another unique experiment directory for a fresh run; never
overwrite this output.

## Files

The runner creates `experiment_config.json`, `outer_fold_results.csv`,
`candidate_search_history.csv`, `threshold_calibrated_fold_results.csv`,
`aggregated_results.csv`, `paired_method_comparisons.csv`,
`lambda_sensitivity.csv`, `runtime_summary.csv`, `completion_log.csv`,
`experiment.log`, `failure_log.txt`, `validation_report.md`, and
`findings_summary.md` here. Result tables distinguish default and calibrated
metrics. Lambda-grid rows represent PSO reruns per lambda, not post-hoc
reselection from one fixed M3 pool.
