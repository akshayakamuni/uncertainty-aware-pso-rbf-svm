# Evaluating Uncertainty-Aware PSO-Tuned SVM Selection on Small Medical Datasets

## Research question

Does penalizing variability in inner-fold loss when guiding PSO search change
outer-fold performance or the inner-to-outer generalization gap for RBF-SVMs
on three small medical benchmark datasets? The saved experiments show
dataset-dependent, mixed results. They do not establish that the method is
universally superior or reliably mitigates overtuning.

## Canonical completed experiments

The current research evidence is in `experiments/`:

- `uncertainty_aware_pso_v2/`: one-seed comparison of four methods, with
  5-fold outer × 5-fold inner stratified CV, 8 particles × 6 iterations, and
  primary lambda 1.
- `uncertainty_aware_pso_v2_multiseed/`: M1 versus M3 at lambda 1 over seeds
  42, 43, and 44, using matched splits and starting particle positions.
- `uncertainty_aware_pso_lambda_multiseed/`: the primary lambda-sensitivity
  evidence, comparing lambda 0, 0.25, 0.5, 1, and 2 over the same three seeds,
  datasets, preprocessing configurations, and nested-CV protocol.
- `expanded_search_v1/run_2/`: a separate larger-budget study comparing
  standard and post-hoc uncertainty-aware selection from the same candidate
  pool. Its PSO optimizes mean loss only; it is not the M3 trajectory method.

All these are saved runs. The documentation and analysis tools do not rerun
model training.

## Methods

For candidate \(j\), let \(L_{jk}=1-\mathrm{AUC}_{jk}\) be its continuous-score
ROC-AUC loss on inner fold \(k\). Let \(\bar L_j\) be the mean and \(s_j\) the
sample standard deviation (`ddof=1`) across the five inner folds.

- **M1 — Standard PSO:** PSO fitness, personal/global-best updates, and final
  selection use \(\bar L_j\).
- **M2 — Post-hoc uncertainty-aware selection:** reselects from the same M1
  candidate pool using \(\bar L_j+\lambda s_j\); it does not change the PSO
  trajectory.
- **M3 — Uncertainty-aware PSO:** uses \(\bar L_j+\lambda s_j\) during
  candidate fitness, personal/global-best updates, and final selection. Its
  trajectory can differ from M1.
- **M4 — Random search:** evaluates an equal-budget set of uniformly sampled
  hyperparameter points and selects by mean loss; it is a separate control,
  not PSO.

Lambda 0 is the standard mean-loss PSO objective. Lambda 1 was the declared
primary uncertainty penalty; 0.25, 0.5, and 2 are sensitivity values. Lambda
is not selected using outer-test outcomes.

## Datasets and preprocessing

The three binary-classification datasets are Wisconsin Diagnostic Breast
Cancer (569 rows, 30 predictors; malignant positive), Pima Indians Diabetes
(768 rows, 8 predictors; selected measurement zeros treated as missing), and
Cleveland Heart Disease (303 rows, 13 predictors; original targets 1–4 mapped
to disease-positive). Four pipelines are evaluated: median imputation,
mean imputation, PCA retaining 95% variance after median imputation, and
five-component TruncatedSVD after median imputation. Imputation, scaling,
encoding, and dimensionality reduction are fitted inside each CV training
partition.

The exact source/version and redistribution terms for the local Pima CSV
remain unverified. Checksums are recorded in `docs/raw_dataset_checksums.md`;
hashes identify file contents but do not establish provenance or license.

## Evaluation protocol

Experiments use nested stratified five-fold outer and inner CV, RBF-SVMs, and
search coordinates `log10(C), log10(gamma) ∈ [-3, 3]`. The completed v2 and
multi-seed comparisons use 8 particles × 6 iterations (48 candidate
evaluations per search). The earlier repository implementation used a
different 6 × 4 budget and is retained as historical code.

Default-threshold metrics use SVC's default prediction threshold. Separately
saved threshold-calibrated metrics choose a threshold maximizing inner
out-of-fold balanced accuracy on outer-training data, then evaluate on the
outer test fold. ROC-AUC uses continuous decision scores. Reported metrics are
ROC-AUC, balanced accuracy, sensitivity, specificity, F1, selected \(C\) and
\(\gamma\), runtime, and the signed inner-to-outer loss gap:

\[
G=(1-\mathrm{outer\ ROC\mbox{-}AUC})-\mathrm{selected\ mean\ inner\ loss}.
\]

Positive and negative gaps are not inherently good or bad; the absolute gap
is reported as a descriptive discrepancy, not proof of overtuning.

## Main saved findings

At lambda 1, M3 versus M1 was mixed across the three seeds. In the paired
multi-seed output, M3 had lower average ROC-AUC, balanced accuracy,
sensitivity, and F1, and slightly higher specificity when averaging the
saved matched observations. The absolute gap was smaller for M3 in 71 of 180
matched comparisons, larger in 75, and equal in 34. These are descriptive
counts, not inferential evidence.

The five-lambda analysis likewise varies by dataset, preprocessing, and
lambda. Its paired fold-level CSVs and generated summary are under
`experiments/uncertainty_aware_pso_lambda_multiseed/`. Do not treat
overlapping folds or repeated seeds on the same datasets as independent
samples. No significance, clinical-benefit, novelty, universal-superiority,
or established overtuning-mitigation claim is supported.

The original v2 run recorded `svd_random_state: null`; SVD results from that
run are not guaranteed reproducible. Later multi-seed runs explicitly seed
SVD per run. Historical outputs were not regenerated.

## Read-only analysis and validation

Generate the final saved-data analysis and figures without fitting models:

```powershell
python deliverables\analyze_final_results.py
```

Run its unit tests:

```powershell
python -m unittest discover -s deliverables -p test_final_analysis.py -v
```

Run experiment validators on existing saved outputs (these do not train
models):

```powershell
python experiments\uncertainty_aware_pso_v2\validate_outputs.py
python experiments\uncertainty_aware_pso_v2_multiseed\validate_multiseed.py
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py
$env:EXPANDED_SEARCH_OUTPUT_DIR = 'experiments\expanded_search_v1\run_2'
python experiments\expanded_search_v1\validate_outputs.py
```

The machine-specific environment used for the submission analysis is recorded
in `deliverables/environment_report.md`. No package versions are represented
as pinned requirements.

## Submission deliverables

- `deliverables/research_paper_final.md`: editable manuscript source using
  verified references and saved results.
- `deliverables/research_presentation_final.pptx`: updated final deck; the
  working-draft presentation is preserved separately.
- `deliverables/final_results_summary.md`: generated data-backed summary.
- `deliverables/figures/`: four 300-dpi lambda, dataset, gap, and threshold
  comparison figures.
- `deliverables/environment_report.md`: observed analysis environment and
  dependency availability.

No final manuscript PDF is included because the environment and project
requirements do not provide a Markdown/PDF renderer.

## Historical code and result tables

The `src/` workflow and root `results/` outputs predate the final trajectory
experiments. In particular, `src/evaluation/posthoc_lambda_sensitivity.py`
reuses the original search pool and is not the final multi-seed M3 lambda
study. These files are retained for audit/history and must not be conflated
with the final seeded evidence.
