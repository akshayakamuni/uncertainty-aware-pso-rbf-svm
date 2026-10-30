# Multi-seed lambda sensitivity findings

Lambda=0 is the standard mean-inner-loss PSO objective (M1). For each nonzero lambda, M3 applies mean loss + lambda × sample SD during PSO fitness and personal/global-best updates, then selects from that lambda-specific candidate pool. All comparisons are paired against lambda=0 by seed, dataset, preprocessing configuration, and outer fold.

## Default-threshold paired differences

Entries are means of the five outer-fold paired differences, averaged across the three seed-level means. Counts are descriptive; no significance test is performed.

| Dataset | Preprocessing | Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ absolute gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| breast_cancer | baseline_median | 0.25 | +0.0001 | -0.0005 | +0.0000 | -0.0009 | -0.0008 | -0.0003 |
| breast_cancer | baseline_median | 0.5 | +0.0005 | +0.0020 | +0.0032 | +0.0009 | +0.0025 | -0.0004 |
| breast_cancer | baseline_median | 1 | +0.0002 | +0.0033 | +0.0048 | +0.0019 | +0.0041 | -0.0001 |
| breast_cancer | baseline_median | 2 | -0.0007 | +0.0021 | +0.0032 | +0.0009 | +0.0025 | +0.0003 |
| breast_cancer | mean_imputation | 0.25 | +0.0001 | -0.0005 | +0.0000 | -0.0009 | -0.0008 | -0.0003 |
| breast_cancer | mean_imputation | 0.5 | +0.0005 | +0.0020 | +0.0032 | +0.0009 | +0.0025 | -0.0004 |
| breast_cancer | mean_imputation | 1 | +0.0002 | +0.0033 | +0.0048 | +0.0019 | +0.0041 | -0.0001 |
| breast_cancer | mean_imputation | 2 | -0.0007 | +0.0021 | +0.0032 | +0.0009 | +0.0025 | +0.0003 |
| breast_cancer | pca_median | 0.25 | -0.0002 | +0.0019 | +0.0047 | -0.0009 | +0.0018 | +0.0000 |
| breast_cancer | pca_median | 0.5 | -0.0007 | +0.0006 | +0.0031 | -0.0019 | +0.0003 | -0.0001 |
| breast_cancer | pca_median | 1 | -0.0005 | +0.0013 | +0.0062 | -0.0037 | +0.0004 | +0.0002 |
| breast_cancer | pca_median | 2 | +0.0003 | +0.0021 | +0.0079 | -0.0037 | +0.0013 | -0.0002 |
| breast_cancer | svd_median | 0.25 | +0.0001 | -0.0002 | +0.0016 | -0.0019 | -0.0007 | +0.0001 |
| breast_cancer | svd_median | 0.5 | +0.0001 | +0.0016 | +0.0031 | +0.0000 | +0.0017 | -0.0002 |
| breast_cancer | svd_median | 1 | -0.0002 | +0.0005 | +0.0000 | +0.0010 | +0.0008 | +0.0001 |
| breast_cancer | svd_median | 2 | -0.0002 | +0.0009 | -0.0000 | +0.0019 | +0.0016 | +0.0001 |
| diabetes | baseline_median | 0.25 | +0.0016 | -0.0127 | -0.0327 | +0.0073 | -0.0346 | +0.0010 |
| diabetes | baseline_median | 0.5 | +0.0013 | +0.0044 | +0.0087 | -0.0000 | +0.0108 | +0.0009 |
| diabetes | baseline_median | 1 | -0.0016 | -0.0026 | -0.0031 | -0.0020 | -0.0045 | -0.0003 |
| diabetes | baseline_median | 2 | +0.0001 | +0.0024 | +0.0048 | -0.0000 | +0.0035 | -0.0012 |
| diabetes | mean_imputation | 0.25 | +0.0006 | +0.0027 | +0.0074 | -0.0020 | +0.0050 | -0.0014 |
| diabetes | mean_imputation | 0.5 | +0.0018 | -0.0133 | -0.0220 | -0.0047 | -0.0358 | -0.0007 |
| diabetes | mean_imputation | 1 | -0.0007 | -0.0065 | -0.0176 | +0.0047 | -0.0157 | +0.0010 |
| diabetes | mean_imputation | 2 | -0.0005 | -0.0364 | -0.0962 | +0.0233 | -0.1087 | +0.0029 |
| diabetes | pca_median | 0.25 | -0.0019 | -0.0092 | -0.0290 | +0.0107 | -0.0336 | +0.0015 |
| diabetes | pca_median | 0.5 | -0.0005 | -0.0243 | -0.0660 | +0.0173 | -0.0761 | +0.0026 |
| diabetes | pca_median | 1 | -0.0008 | -0.0369 | -0.1012 | +0.0273 | -0.1151 | +0.0021 |
| diabetes | pca_median | 2 | -0.0014 | -0.0334 | -0.0822 | +0.0153 | -0.0905 | +0.0033 |
| diabetes | svd_median | 0.25 | +0.0002 | +0.0054 | +0.0068 | +0.0040 | +0.0087 | +0.0014 |
| diabetes | svd_median | 0.5 | -0.0005 | -0.0074 | -0.0214 | +0.0067 | -0.0209 | +0.0005 |
| diabetes | svd_median | 1 | -0.0005 | +0.0073 | +0.0020 | +0.0127 | +0.0206 | +0.0014 |
| diabetes | svd_median | 2 | +0.0000 | +0.0122 | +0.0270 | -0.0027 | +0.0368 | +0.0011 |
| heart_disease | baseline_median | 0.25 | -0.0006 | -0.0014 | -0.0049 | +0.0020 | -0.0018 | -0.0005 |
| heart_disease | baseline_median | 0.5 | -0.0004 | -0.0015 | -0.0071 | +0.0041 | -0.0023 | +0.0001 |
| heart_disease | baseline_median | 1 | -0.0011 | -0.0048 | -0.0095 | -0.0000 | -0.0063 | -0.0012 |
| heart_disease | baseline_median | 2 | -0.0083 | -0.0196 | -0.0453 | +0.0061 | -0.0407 | -0.0029 |
| heart_disease | mean_imputation | 0.25 | -0.0006 | -0.0014 | -0.0049 | +0.0020 | -0.0018 | -0.0005 |
| heart_disease | mean_imputation | 0.5 | -0.0004 | -0.0015 | -0.0071 | +0.0041 | -0.0023 | +0.0001 |
| heart_disease | mean_imputation | 1 | -0.0011 | -0.0048 | -0.0095 | -0.0000 | -0.0063 | -0.0012 |
| heart_disease | mean_imputation | 2 | -0.0083 | -0.0196 | -0.0453 | +0.0061 | -0.0407 | -0.0029 |
| heart_disease | pca_median | 0.25 | +0.0014 | +0.0022 | +0.0023 | +0.0021 | +0.0022 | -0.0010 |
| heart_disease | pca_median | 0.5 | +0.0015 | -0.0005 | -0.0053 | +0.0042 | -0.0026 | -0.0012 |
| heart_disease | pca_median | 1 | -0.0002 | -0.0320 | -0.0640 | +0.0000 | -0.0658 | -0.0009 |
| heart_disease | pca_median | 2 | -0.0093 | -0.0510 | -0.1021 | +0.0001 | -0.1121 | +0.0004 |
| heart_disease | svd_median | 0.25 | -0.0018 | +0.0098 | +0.0358 | -0.0162 | +0.0405 | -0.0001 |
| heart_disease | svd_median | 0.5 | +0.0002 | -0.0075 | -0.0049 | -0.0101 | -0.0070 | -0.0010 |
| heart_disease | svd_median | 1 | +0.0032 | -0.0018 | +0.0105 | -0.0141 | +0.0013 | +0.0026 |
| heart_disease | svd_median | 2 | -0.0015 | -0.0616 | -0.1334 | +0.0102 | -0.1387 | +0.0021 |

## Threshold-calibrated paired differences

Threshold calibration is separate and uses only outer-training inner out-of-fold decision scores. The outer-test fold is used only for evaluation.

| Dataset | Preprocessing | Lambda | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | baseline_median | 0.25 | +0.0001 | +0.0006 | -0.0016 | +0.0028 | +0.0014 |
| breast_cancer | baseline_median | 0.5 | +0.0005 | +0.0006 | -0.0063 | +0.0075 | +0.0027 |
| breast_cancer | baseline_median | 1 | +0.0002 | -0.0007 | -0.0079 | +0.0065 | +0.0011 |
| breast_cancer | baseline_median | 2 | -0.0007 | -0.0035 | -0.0079 | +0.0010 | -0.0033 |
| breast_cancer | mean_imputation | 0.25 | +0.0001 | +0.0006 | -0.0016 | +0.0028 | +0.0014 |
| breast_cancer | mean_imputation | 0.5 | +0.0005 | +0.0006 | -0.0063 | +0.0075 | +0.0027 |
| breast_cancer | mean_imputation | 1 | +0.0002 | -0.0007 | -0.0079 | +0.0065 | +0.0011 |
| breast_cancer | mean_imputation | 2 | -0.0007 | -0.0035 | -0.0079 | +0.0010 | -0.0033 |
| breast_cancer | pca_median | 0.25 | -0.0002 | -0.0042 | -0.0046 | -0.0037 | -0.0055 |
| breast_cancer | pca_median | 0.5 | -0.0007 | -0.0034 | -0.0031 | -0.0037 | -0.0045 |
| breast_cancer | pca_median | 1 | -0.0005 | -0.0051 | -0.0047 | -0.0055 | -0.0069 |
| breast_cancer | pca_median | 2 | +0.0003 | -0.0079 | +0.0001 | -0.0158 | -0.0123 |
| breast_cancer | svd_median | 0.25 | +0.0001 | -0.0027 | -0.0064 | +0.0009 | -0.0029 |
| breast_cancer | svd_median | 0.5 | +0.0001 | -0.0037 | -0.0065 | -0.0009 | -0.0044 |
| breast_cancer | svd_median | 1 | -0.0002 | -0.0039 | -0.0049 | -0.0028 | -0.0050 |
| breast_cancer | svd_median | 2 | -0.0002 | -0.0049 | -0.0080 | -0.0019 | -0.0058 |
| diabetes | baseline_median | 0.25 | +0.0016 | -0.0001 | -0.0175 | +0.0173 | -0.0007 |
| diabetes | baseline_median | 0.5 | +0.0013 | -0.0005 | +0.0063 | -0.0073 | +0.0003 |
| diabetes | baseline_median | 1 | -0.0016 | +0.0023 | +0.0012 | +0.0033 | +0.0039 |
| diabetes | baseline_median | 2 | +0.0001 | -0.0010 | +0.0160 | -0.0180 | -0.0001 |
| diabetes | mean_imputation | 0.25 | +0.0006 | +0.0002 | +0.0163 | -0.0160 | +0.0004 |
| diabetes | mean_imputation | 0.5 | +0.0018 | +0.0062 | +0.0424 | -0.0300 | +0.0074 |
| diabetes | mean_imputation | 1 | -0.0007 | +0.0017 | +0.0100 | -0.0067 | +0.0038 |
| diabetes | mean_imputation | 2 | -0.0005 | +0.0020 | +0.0313 | -0.0273 | +0.0041 |
| diabetes | pca_median | 0.25 | -0.0019 | +0.0024 | +0.0262 | -0.0213 | +0.0038 |
| diabetes | pca_median | 0.5 | -0.0005 | +0.0022 | +0.0224 | -0.0180 | +0.0036 |
| diabetes | pca_median | 1 | -0.0008 | -0.0021 | +0.0124 | -0.0167 | -0.0015 |
| diabetes | pca_median | 2 | -0.0014 | +0.0026 | +0.0152 | -0.0100 | +0.0039 |
| diabetes | svd_median | 0.25 | +0.0002 | -0.0035 | +0.0024 | -0.0093 | -0.0031 |
| diabetes | svd_median | 0.5 | -0.0005 | -0.0003 | +0.0273 | -0.0280 | +0.0019 |
| diabetes | svd_median | 1 | -0.0005 | +0.0073 | +0.0260 | -0.0113 | +0.0102 |
| diabetes | svd_median | 2 | +0.0000 | +0.0059 | +0.0284 | -0.0167 | +0.0085 |
| heart_disease | baseline_median | 0.25 | -0.0006 | +0.0038 | +0.0096 | -0.0020 | +0.0052 |
| heart_disease | baseline_median | 0.5 | -0.0004 | +0.0019 | +0.0119 | -0.0081 | +0.0033 |
| heart_disease | baseline_median | 1 | -0.0011 | +0.0094 | +0.0168 | +0.0020 | +0.0114 |
| heart_disease | baseline_median | 2 | -0.0083 | -0.0023 | +0.0215 | -0.0262 | +0.0015 |
| heart_disease | mean_imputation | 0.25 | -0.0006 | +0.0038 | +0.0096 | -0.0020 | +0.0052 |
| heart_disease | mean_imputation | 0.5 | -0.0004 | +0.0019 | +0.0119 | -0.0081 | +0.0033 |
| heart_disease | mean_imputation | 1 | -0.0011 | +0.0094 | +0.0168 | +0.0020 | +0.0114 |
| heart_disease | mean_imputation | 2 | -0.0083 | -0.0023 | +0.0215 | -0.0262 | +0.0015 |
| heart_disease | pca_median | 0.25 | +0.0014 | -0.0002 | -0.0046 | +0.0042 | -0.0014 |
| heart_disease | pca_median | 0.5 | +0.0015 | -0.0067 | -0.0072 | -0.0061 | -0.0080 |
| heart_disease | pca_median | 1 | -0.0002 | +0.0005 | +0.0027 | -0.0018 | -0.0006 |
| heart_disease | pca_median | 2 | -0.0093 | -0.0091 | +0.0122 | -0.0303 | -0.0080 |
| heart_disease | svd_median | 0.25 | -0.0018 | -0.0115 | -0.0190 | -0.0040 | -0.0143 |
| heart_disease | svd_median | 0.5 | +0.0002 | -0.0132 | -0.0142 | -0.0122 | -0.0146 |
| heart_disease | svd_median | 1 | +0.0032 | -0.0168 | -0.0213 | -0.0122 | -0.0195 |
| heart_disease | svd_median | 2 | -0.0015 | -0.0085 | +0.0095 | -0.0265 | -0.0056 |

## Runtime

| Seed | Elapsed seconds | Elapsed hours |
|---:|---:|---:|
| 42 | 1680.94 | 0.47 |
| 43 | 2079.09 | 0.58 |
| 44 | 4215.05 | 1.17 |
| Total sequential | 7975.08 | 2.22 |

## Interpretation limits

- Outer folds overlap; folds and seeds on the same datasets are not independent observations.
- These are descriptive comparisons; no statistical significance is claimed.
- This experiment does not establish universal superiority, reduced overtuning, clinical benefit, or novelty.
- Only the three declared seeds and three datasets were tested.
- Lambda changes the PSO objective and its best updates; it is not a final-selection-only post-hoc comparison.
- Default-threshold and calibrated outputs are stored separately.
