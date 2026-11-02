# Final saved-results analysis

Generated from the saved final lambda fold-level CSVs and the completed three-seed M1/M3 fold outputs. The script does not fit models.

## Protocol and aggregation

Lambda 0 is M1's mean-inner-loss PSO objective; positive lambdas are M3 runs whose penalized fitness also guides personal/global-best updates. Each lambda comparison is paired by seed, dataset, preprocessing, and outer fold. Fold-level differences are retained, then averaged within seed before summarizing across the three seeds. Values below are descriptive; overlapping folds and reuse of the same datasets mean these rows are not independent samples.

## Mean paired changes from lambda=0

Default-threshold metrics and the signed/absolute gap use the complete matched lambda run data. Each table entry averages fold- and preprocessing-level differences within seed, then the three seed means.

| Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ absolute gap |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | -0.0001 | -0.0003 | -0.0011 | +0.0004 | -0.0013 | -0.0000 |
| 0.5 | +0.0003 | -0.0038 | -0.0094 | +0.0018 | -0.0108 | +0.0000 |
| 1 | -0.0003 | -0.0061 | -0.0147 | +0.0025 | -0.0152 | +0.0003 |
| 2 | -0.0025 | -0.0167 | -0.0382 | +0.0049 | -0.0403 | +0.0003 |

## Dataset-level paired changes

| Dataset | Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ absolute gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | 0.25 | +0.0000 | +0.0002 | +0.0016 | -0.0012 | -0.0001 | -0.0001 |
| breast_cancer | 0.5 | +0.0001 | +0.0016 | +0.0031 | -0.0000 | +0.0018 | -0.0003 |
| breast_cancer | 1 | -0.0001 | +0.0021 | +0.0039 | +0.0002 | +0.0024 | +0.0000 |
| breast_cancer | 2 | -0.0003 | +0.0018 | +0.0036 | +0.0000 | +0.0020 | +0.0001 |
| diabetes | 0.25 | +0.0001 | -0.0034 | -0.0119 | +0.0050 | -0.0136 | +0.0006 |
| diabetes | 0.5 | +0.0005 | -0.0102 | -0.0252 | +0.0048 | -0.0305 | +0.0008 |
| diabetes | 1 | -0.0009 | -0.0097 | -0.0300 | +0.0107 | -0.0287 | +0.0010 |
| diabetes | 2 | -0.0004 | -0.0138 | -0.0366 | +0.0090 | -0.0397 | +0.0015 |
| heart_disease | 0.25 | -0.0004 | +0.0023 | +0.0071 | -0.0025 | +0.0098 | -0.0005 |
| heart_disease | 0.5 | +0.0003 | -0.0028 | -0.0061 | +0.0006 | -0.0035 | -0.0005 |
| heart_disease | 1 | +0.0002 | -0.0108 | -0.0181 | -0.0035 | -0.0193 | -0.0002 |
| heart_disease | 2 | -0.0068 | -0.0380 | -0.0815 | +0.0056 | -0.0830 | -0.0008 |

## Threshold modes

Threshold-calibrated results use the saved outer-training inner-OOF thresholds and remain distinct from default-threshold results. Calibration changes classification metrics, not continuous-score ROC-AUC. The table averages dataset/configuration/fold rows within each seed, then reports the mean of the three seed means.

| Lambda | Threshold mode | Balanced accuracy | Sensitivity | Specificity | F1 |
| --- | --- | --- | --- | --- | --- |
| 0 | Default | +0.8216 | +0.7186 | +0.9246 | +0.7554 |
| 0.25 | Default | +0.8213 | +0.7176 | +0.9251 | +0.7541 |
| 0.5 | Default | +0.8178 | +0.7093 | +0.9264 | +0.7447 |
| 1 | Default | +0.8155 | +0.7039 | +0.9271 | +0.7402 |
| 2 | Default | +0.8050 | +0.6804 | +0.9295 | +0.7152 |
| 0 | Calibrated | +0.8439 | +0.8470 | +0.8408 | +0.8113 |
| 0.25 | Calibrated | +0.8430 | +0.8478 | +0.8382 | +0.8105 |
| 0.5 | Calibrated | +0.8427 | +0.8536 | +0.8318 | +0.8108 |
| 1 | Calibrated | +0.8440 | +0.8503 | +0.8377 | +0.8121 |
| 2 | Calibrated | +0.8412 | +0.8580 | +0.8244 | +0.8098 |

## Three-seed M1 versus M3 at lambda=1

Paired average M3-minus-M1 differences across matched saved rows: outer_roc_auc -0.0003; balanced_accuracy -0.0061; sensitivity -0.0147; specificity +0.0025; f1_score -0.0152.

The absolute inner-to-outer gap was smaller for M3 in 71 of 180 matched comparisons, larger in 75, and equal in 34. This is not consistent gap reduction.

## Interpretation and limitations

The saved results show changes that depend on dataset, preprocessing, lambda, and threshold mode; they do not establish that M3 is better than M1 or that a nonzero penalty reliably reduces overtuning. No p-values, confidence intervals, clinical-benefit claims, or claims of statistical significance are produced. The three seeds reuse the same datasets, and overlapping CV folds are not treated as independent.

Historical v2 TruncatedSVD used `random_state: null`; these historical outputs were not regenerated. The final seeded lambda runs record the SVD random state per seed. The exact provenance/version and license terms for the local Pima CSV remain unresolved.

## Figure files


- `figures/lambda_metric_changes.png`
- `figures/dataset_lambda_comparisons.png`
- `figures/absolute_generalization_gap.png`
- `figures/threshold_mode_comparison.png`

The detailed paired rows recomputed from raw outputs are in `lambda_paired_differences_recomputed.csv`; seed-level summaries are in `lambda_seed_dataset_preprocessing_summary.csv`.
