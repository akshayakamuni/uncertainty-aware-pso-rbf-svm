# Results Audit and Interpretation

## Source-of-truth hierarchy

The completed and validated experiment outputs are preserved under
`experiments/`. The latest lambda-sensitivity evidence is in
`experiments/uncertainty_aware_pso_lambda_multiseed/`, with seed-level
default-threshold and threshold-calibrated fold CSVs and paired
lambda-minus-zero comparisons. The three-seed M1/M3 comparison is in
`experiments/uncertainty_aware_pso_v2_multiseed/`. The one-seed, four-method
v2 result is in `experiments/uncertainty_aware_pso_v2/`.

The old root `results/` tables and `src/evaluation/posthoc_lambda_sensitivity.py`
are historical. That lambda analysis reselects candidates from the original
mean-loss search pool; it does not rerun the penalized PSO trajectory for
each lambda. Do not combine those outputs with the final seeded M3
lambda-sensitivity experiment.

## Method comparison

- M1: Standard PSO, mean inner loss for fitness, best updates, and selection.
- M2: Post-hoc uncertainty-aware selection from the M1 candidate pool.
- M3: PSO fitness, personal/global-best updates, and final selection use mean
  loss plus lambda times sample SD (`ddof=1`).
- M4: Equal-budget uniform random search, selected by mean inner loss.

Lambda 0 is the standard PSO objective. Lambda 1 was the primary uncertainty
penalty; 0.25, 0.5, and 2 are sensitivity values.

## Completed studies and validation

The v2, three-seed M1/M3, final five-lambda, and expanded-search `run_2`
outputs have saved validation reports. The lambda multi-seed study includes
separate default and threshold-calibrated results and pairs each nonzero
lambda against zero by seed, dataset, preprocessing configuration, and outer
fold. The analysis script in `deliverables/analyze_final_results.py`
recomputes summaries from those saved fold files and verifies the saved
paired lambda tables without fitting models.

The final M1/M3 comparison is mixed. Averaged over the saved matched
observations, M3-minus-M1 differences are negative for ROC-AUC, balanced
accuracy, sensitivity, and F1, with a small positive difference for
specificity. They vary by dataset and preprocessing. In 180 matched
comparisons, M3's absolute inner-to-outer gap is smaller in 71, larger in 75,
and equal in 34. This does not support a consistent gap reduction or a claim
that uncertainty-aware PSO mitigates overtuning.

The final lambda experiment also shows dataset- and lambda-dependent trade
offs. Lambda sensitivity must be interpreted from its own seeded M3
trajectories and not from the historical post-hoc table. The three seeds reuse
the same datasets and the outer folds overlap; summaries are descriptive, not
independent samples. No significance, universal-superiority, clinical
benefit, or novelty claim is made.

## Metric definitions

Inner loss is \(1-\mathrm{ROC\mbox{-}AUC}\) from continuous decision scores.
The signed generalization gap is
\[
G=(1-\mathrm{outer\ ROC\mbox{-}AUC})-\mathrm{selected\ mean\ inner\ loss}.
\]
Absolute gaps are reported as descriptive distances. A negative signed gap
does not imply better generalization, and a smaller absolute gap is not by
itself proof of less overtuning.

Default-threshold classification metrics use SVC predictions. Calibrated
classification metrics use a threshold selected by outer-training inner OOF
balanced accuracy and are reported separately. Because that threshold is
applied after refitting on all outer-training rows, a score-scale mismatch is
possible.

## Historical reproducibility and table discrepancies

The original v2 config records `svd_random_state: null`; its historical SVD
outputs are not guaranteed reproducible. Later v2 source and completed
multi-seed runs explicitly seed TruncatedSVD. The original outputs were not
regenerated.

An older root `results/` default/calibrated comparison has a known
Heart-Disease, SVD-median, outer-fold-4 Standard candidate mismatch. A
consistency-checked copy is retained under
`results/additional_analysis/threshold_calibrated_consistency_checked_outer_folds.csv`.
Separately, v2 default and calibrated AUC differ on Heart-Disease,
SVD-median, outer fold 1 for M1, despite matching C/gamma; the historical
unseeded SVD is a reproducibility concern. In the final seeded lambda
experiment, selected candidate IDs and AUC match between threshold modes;
threshold-dependent classification metrics are separate by design. Do not
silently reconcile or overwrite historical tables.

## Data and claims limitations


Raw dataset checksums are recorded in `docs/raw_dataset_checksums.md`. Exact
provenance/version and license terms for the local Pima CSV remain
unverified; licenses/redistribution terms for all local data should be
confirmed before release. The Pima zero-as-missing rule is a modeling
assumption. Three datasets and three random seeds do not establish broad
robustness. This is not a clinical validation study.
