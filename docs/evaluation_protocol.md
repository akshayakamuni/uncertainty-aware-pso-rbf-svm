# Evaluation Protocol

## Objective and method definitions

The experiments assess whether variability-penalized candidate fitness is
associated with changes in outer-fold performance or the inner-to-outer loss
gap. This is an empirical research question, not an assumption that
uncertainty-aware search is superior.

For candidate \(j\) and inner fold \(k\), define
\(L_{jk}=1-\mathrm{ROC\mbox{-}AUC}_{jk}\), where AUC is computed from
continuous decision scores. Across five inner folds, calculate mean loss
\(\bar L_j\) and sample standard deviation \(s_j\) (`ddof=1`). The uncertainty
criterion is
\[
J_\lambda(j)=\bar L_j+\lambda s_j.
\]

- **M1 Standard PSO:** PSO fitness and personal/global-best updates minimize
  mean inner loss; final selection also minimizes mean loss.
- **M2 Post-hoc uncertainty-aware selection:** selects with \(J_1\) from the
  M1 recorded candidate pool; the swarm trajectory remains unchanged.
- **M3 Uncertainty-aware PSO:** evaluates candidates and updates both
  personal/global bests with \(J_\lambda\), then selects from that
  lambda-specific search history. Nonzero lambda can therefore change the
  PSO trajectory, not only final selection.
- **M4 Random search:** uniformly samples 48 points within the search bounds
  and selects by mean inner loss. It is not a PSO variant.

Lambda 0 is equivalent to the standard mean-loss PSO objective. Lambda 1 is
the predeclared primary penalty. Values 0.25, 0.5, and 2 are sensitivity
analyses and are not selected by outer-test performance.

## Datasets, targets, and preprocessing

The three datasets are Wisconsin Diagnostic Breast Cancer, Pima Indians
Diabetes, and Cleveland Heart Disease. Target mappings and the Diabetes
zero-as-missing assumption are described in `dataset_audit.md` and
`preprocessing_protocol.md`.

The four configurations are median imputation, mean imputation, PCA retaining
95% variance after median imputation, and five-component TruncatedSVD after
median imputation. Numeric features are imputed and standardized; categorical
Heart Disease features are imputed and one-hot encoded. Every transform is in
the estimator pipeline and is fitted only on the current training partition.

## Nested cross-validation and search

Use shuffled, stratified 5-fold outer CV and stratified 5-fold inner CV within
each outer-training partition. RBF-SVM search parameters are
\(\log_{10}(C)\) and \(\log_{10}(\gamma)\), each in \([-3,3]\). Completed v2,
M1/M3 multi-seed, and lambda multi-seed studies use 8 particles × 6 iterations
(48 candidate evaluations per search), with inertia 0.7 and cognitive/social
coefficients 1.5. Seeds 42, 43, and 44 are used in the multi-seed studies.
For each matched dataset/configuration/fold/seed, comparisons share the same
outer/inner split plan and initial PSO particle coordinates where applicable.

The earlier `src/evaluation/nested_cv_experiment.py` used 6 particles × 4
iterations. It is retained as historical implementation and is not the
configuration for the final v2 or multi-seed results.

## Leakage controls and thresholds

All tuning uses only outer-training data and inner splits. The selected model
is refit on the complete outer-training partition and scored on the untouched
outer test fold. Outer-test labels are not used in hyperparameter, lambda, or
threshold selection.

Default-threshold and threshold-calibrated outputs are separate. Default
classification metrics use SVC predictions. For calibration, inner
out-of-fold decision scores from the outer-training partition are used to
choose the threshold maximizing balanced accuracy; that threshold is applied
to outer-test decision scores after refitting on all outer-training rows.
ROC-AUC always uses continuous outer-test decision scores and should be
unchanged by threshold alone. The calibration procedure may have a score-scale
mismatch between inner-fold models and the refitted model.

## Metrics and generalization gap

Report ROC-AUC, balanced accuracy, sensitivity, specificity, F1, selected
\(C,\gamma\), candidate evaluations, runtime, mean/sample SD inner loss, and
the signed gap
\[
G=(1-\mathrm{outer\ ROC\mbox{-}AUC})-\bar L_j.
\]
The absolute gap \(|G|\) is also used descriptively. A negative signed gap is
not automatically better; neither signed nor absolute gap alone proves
overtuning.

## Aggregation and inference limits

Lambda comparisons are paired at seed, dataset, preprocessing configuration,
and outer-fold level against lambda 0. First compute these matched differences;
then summarize across folds/configurations within each seed and report the
variation among the three seed-level summaries. Default-threshold and
calibrated metrics remain separate. Do not treat overlapping folds or seeds
using the same datasets as independent samples. The saved studies do not
provide a basis for a significance claim, population generalization claim, or
clinical-benefit claim.

## Reproducibility caveats

The original v2 historical run used `TruncatedSVD(random_state=None)`, as
recorded in its saved config. The v2 runner was subsequently corrected to set
the SVD random state from its configured seed. The completed M1/M3 and
five-lambda multi-seed runs record and use a per-run SVD seed. The historical
v2 outputs were not regenerated; do not describe them as independently
reproduced.

The local Diabetes CSV's exact source/version and licensing terms for local
data files remain unresolved. Checksums identify file contents but do not
prove provenance or redistribution permissions.
