# Multi-seed M1 vs M3 findings

M1 selects from PSO candidates by mean inner-fold loss. M3 runs PSO with mean loss + lambda times sample SD as fitness and uses lambda=1. M3-minus-M1 differences are paired by seed, dataset, preprocessing configuration, and outer fold.

## Mean outer-fold metrics by dataset and preprocessing

| Dataset | Preprocessing | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| breast_cancer | baseline_median | M1 Standard PSO | 0.9954 | 0.9693 | 0.9497 | 0.9888 | 0.9646 |
| breast_cancer | baseline_median | M3 UA-PSO | 0.9955 | 0.9726 | 0.9545 | 0.9907 | 0.9687 |
| breast_cancer | mean_imputation | M1 Standard PSO | 0.9954 | 0.9693 | 0.9497 | 0.9888 | 0.9646 |
| breast_cancer | mean_imputation | M3 UA-PSO | 0.9955 | 0.9726 | 0.9545 | 0.9907 | 0.9687 |
| breast_cancer | pca_median | M1 Standard PSO | 0.9953 | 0.9669 | 0.9450 | 0.9888 | 0.9620 |
| breast_cancer | pca_median | M3 UA-PSO | 0.9949 | 0.9682 | 0.9513 | 0.9851 | 0.9624 |
| breast_cancer | svd_median | M1 Standard PSO | 0.9952 | 0.9585 | 0.9403 | 0.9767 | 0.9499 |
| breast_cancer | svd_median | M3 UA-PSO | 0.9951 | 0.9590 | 0.9403 | 0.9776 | 0.9507 |
| diabetes | baseline_median | M1 Standard PSO | 0.8372 | 0.6977 | 0.5034 | 0.8920 | 0.5730 |
| diabetes | baseline_median | M3 UA-PSO | 0.8356 | 0.6951 | 0.5003 | 0.8900 | 0.5685 |
| diabetes | mean_imputation | M1 Standard PSO | 0.8369 | 0.7085 | 0.5310 | 0.8860 | 0.6052 |
| diabetes | mean_imputation | M3 UA-PSO | 0.8362 | 0.7020 | 0.5134 | 0.8907 | 0.5895 |
| diabetes | pca_median | M1 Standard PSO | 0.8391 | 0.7119 | 0.5410 | 0.8827 | 0.6142 |
| diabetes | pca_median | M3 UA-PSO | 0.8383 | 0.6749 | 0.4398 | 0.9100 | 0.4991 |
| diabetes | svd_median | M1 Standard PSO | 0.8148 | 0.6252 | 0.3464 | 0.9040 | 0.3867 |
| diabetes | svd_median | M3 UA-PSO | 0.8143 | 0.6325 | 0.3484 | 0.9167 | 0.4073 |
| heart_disease | baseline_median | M1 Standard PSO | 0.9138 | 0.8452 | 0.7961 | 0.8943 | 0.8279 |
| heart_disease | baseline_median | M3 UA-PSO | 0.9127 | 0.8405 | 0.7866 | 0.8943 | 0.8215 |
| heart_disease | mean_imputation | M1 Standard PSO | 0.9138 | 0.8452 | 0.7961 | 0.8943 | 0.8279 |
| heart_disease | mean_imputation | M3 UA-PSO | 0.9127 | 0.8405 | 0.7866 | 0.8943 | 0.8215 |
| heart_disease | pca_median | M1 Standard PSO | 0.9155 | 0.8421 | 0.8083 | 0.8760 | 0.8260 |
| heart_disease | pca_median | M3 UA-PSO | 0.9153 | 0.8101 | 0.7443 | 0.8760 | 0.7603 |
| heart_disease | svd_median | M1 Standard PSO | 0.8700 | 0.7197 | 0.5166 | 0.9229 | 0.5631 |
| heart_disease | svd_median | M3 UA-PSO | 0.8732 | 0.7179 | 0.5271 | 0.9088 | 0.5644 |

## Paired M3 minus M1 differences by seed, dataset, and preprocessing

Values below are means of the five matched outer-fold differences for each seed/dataset/preprocessing combination. The fold-level paired table is in `paired_fold_differences.csv`.

| Seed | Dataset | Preprocessing | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | breast_cancer | baseline_median | +0.0011 | +0.0014 | +0.0000 | +0.0028 | +0.0023 |
| 42 | breast_cancer | mean_imputation | +0.0011 | +0.0014 | +0.0000 | +0.0028 | +0.0023 |
| 42 | breast_cancer | pca_median | -0.0022 | +0.0037 | +0.0186 | -0.0112 | +0.0013 |
| 42 | breast_cancer | svd_median | +0.0006 | +0.0023 | +0.0047 | +0.0000 | +0.0026 |
| 42 | diabetes | baseline_median | +0.0005 | +0.0011 | -0.0037 | +0.0060 | +0.0007 |
| 42 | diabetes | mean_imputation | -0.0002 | +0.0037 | +0.0034 | +0.0040 | +0.0081 |
| 42 | diabetes | pca_median | +0.0024 | -0.0371 | -0.0962 | +0.0220 | -0.1144 |
| 42 | diabetes | svd_median | +0.0028 | -0.0167 | -0.0815 | +0.0480 | -0.0848 |
| 42 | heart_disease | baseline_median | +0.0013 | +0.0004 | +0.0071 | -0.0063 | +0.0006 |
| 42 | heart_disease | mean_imputation | +0.0013 | +0.0004 | +0.0071 | -0.0063 | +0.0006 |
| 42 | heart_disease | pca_median | +0.0037 | +0.0005 | +0.0071 | -0.0061 | +0.0011 |
| 42 | heart_disease | svd_median | +0.0039 | +0.0665 | +0.1571 | -0.0242 | +0.1633 |
| 43 | breast_cancer | baseline_median | +0.0003 | +0.0062 | +0.0095 | +0.0028 | +0.0076 |
| 43 | breast_cancer | mean_imputation | +0.0003 | +0.0062 | +0.0095 | +0.0028 | +0.0076 |
| 43 | breast_cancer | pca_median | -0.0000 | -0.0004 | +0.0048 | -0.0056 | -0.0019 |
| 43 | breast_cancer | svd_median | -0.0007 | -0.0009 | -0.0047 | +0.0028 | -0.0003 |
| 43 | diabetes | baseline_median | +0.0019 | +0.0321 | +0.0943 | -0.0300 | +0.1073 |
| 43 | diabetes | mean_imputation | -0.0009 | -0.0196 | -0.0451 | +0.0060 | -0.0492 |
| 43 | diabetes | pca_median | -0.0049 | -0.0385 | -0.1131 | +0.0360 | -0.1190 |
| 43 | diabetes | svd_median | -0.0025 | +0.0488 | +0.1155 | -0.0180 | +0.1637 |
| 43 | heart_disease | baseline_median | -0.0030 | -0.0107 | -0.0214 | +0.0000 | -0.0130 |
| 43 | heart_disease | mean_imputation | -0.0030 | -0.0107 | -0.0214 | +0.0000 | -0.0130 |
| 43 | heart_disease | pca_median | -0.0008 | -0.0137 | -0.0214 | -0.0061 | -0.0163 |
| 43 | heart_disease | svd_median | +0.0033 | -0.0182 | +0.0000 | -0.0364 | -0.0158 |
| 44 | breast_cancer | baseline_median | -0.0009 | +0.0024 | +0.0048 | +0.0000 | +0.0025 |
| 44 | breast_cancer | mean_imputation | -0.0009 | +0.0024 | +0.0048 | +0.0000 | +0.0025 |
| 44 | breast_cancer | pca_median | +0.0008 | +0.0005 | -0.0047 | +0.0056 | +0.0018 |
| 44 | breast_cancer | svd_median | -0.0004 | +0.0000 | +0.0000 | +0.0000 | +0.0001 |
| 44 | diabetes | baseline_median | -0.0071 | -0.0410 | -0.1000 | +0.0180 | -0.1214 |
| 44 | diabetes | mean_imputation | -0.0010 | -0.0036 | -0.0111 | +0.0040 | -0.0061 |
| 44 | diabetes | pca_median | +0.0002 | -0.0352 | -0.0943 | +0.0240 | -0.1118 |
| 44 | diabetes | svd_median | -0.0019 | -0.0100 | -0.0280 | +0.0080 | -0.0170 |
| 44 | heart_disease | baseline_median | -0.0015 | -0.0040 | -0.0143 | +0.0062 | -0.0066 |
| 44 | heart_disease | mean_imputation | -0.0015 | -0.0040 | -0.0143 | +0.0062 | -0.0066 |
| 44 | heart_disease | pca_median | -0.0037 | -0.0828 | -0.1778 | +0.0121 | -0.1821 |
| 44 | heart_disease | svd_median | +0.0025 | -0.0537 | -0.1257 | +0.0182 | -0.1436 |

## Direction consistency

Counts are over the 36 seed × dataset × preprocessing mean contrasts (not independent inferential observations).

| Metric | M3 higher | M3 lower | Equal | Improvement consistent in all groups? |
|---|---:|---:|---:|---|
| outer_roc_auc | 17 | 19 | 0 | No |
| balanced_accuracy | 18 | 18 | 0 | No |
| sensitivity | 15 | 18 | 3 | No |
| specificity | 21 | 10 | 5 | No |
| f1_score | 18 | 18 | 0 | No |

## Across-seed variation of paired means

Mean and sample SD are computed over the three seed-level means for each dataset/preprocessing pair.

| Dataset | Preprocessing | Metric | Mean M3−M1 | SD across seeds |
|---|---|---|---:|---:|
| breast_cancer | baseline_median | outer_roc_auc | +0.0002 | 0.0010 |
| breast_cancer | baseline_median | balanced_accuracy | +0.0033 | 0.0025 |
| breast_cancer | baseline_median | sensitivity | +0.0048 | 0.0048 |
| breast_cancer | baseline_median | specificity | +0.0019 | 0.0016 |
| breast_cancer | baseline_median | f1_score | +0.0041 | 0.0030 |
| breast_cancer | mean_imputation | outer_roc_auc | +0.0002 | 0.0010 |
| breast_cancer | mean_imputation | balanced_accuracy | +0.0033 | 0.0025 |
| breast_cancer | mean_imputation | sensitivity | +0.0048 | 0.0048 |
| breast_cancer | mean_imputation | specificity | +0.0019 | 0.0016 |
| breast_cancer | mean_imputation | f1_score | +0.0041 | 0.0030 |
| breast_cancer | pca_median | outer_roc_auc | -0.0005 | 0.0015 |
| breast_cancer | pca_median | balanced_accuracy | +0.0013 | 0.0022 |
| breast_cancer | pca_median | sensitivity | +0.0062 | 0.0117 |
| breast_cancer | pca_median | specificity | -0.0037 | 0.0085 |
| breast_cancer | pca_median | f1_score | +0.0004 | 0.0020 |
| breast_cancer | svd_median | outer_roc_auc | -0.0002 | 0.0007 |
| breast_cancer | svd_median | balanced_accuracy | +0.0005 | 0.0017 |
| breast_cancer | svd_median | sensitivity | +0.0000 | 0.0047 |
| breast_cancer | svd_median | specificity | +0.0010 | 0.0016 |
| breast_cancer | svd_median | f1_score | +0.0008 | 0.0016 |
| diabetes | baseline_median | outer_roc_auc | -0.0016 | 0.0048 |
| diabetes | baseline_median | balanced_accuracy | -0.0026 | 0.0367 |
| diabetes | baseline_median | sensitivity | -0.0031 | 0.0971 |
| diabetes | baseline_median | specificity | -0.0020 | 0.0250 |
| diabetes | baseline_median | f1_score | -0.0045 | 0.1145 |
| diabetes | mean_imputation | outer_roc_auc | -0.0007 | 0.0004 |
| diabetes | mean_imputation | balanced_accuracy | -0.0065 | 0.0119 |
| diabetes | mean_imputation | sensitivity | -0.0176 | 0.0249 |
| diabetes | mean_imputation | specificity | +0.0047 | 0.0012 |
| diabetes | mean_imputation | f1_score | -0.0157 | 0.0298 |
| diabetes | pca_median | outer_roc_auc | -0.0008 | 0.0037 |
| diabetes | pca_median | balanced_accuracy | -0.0369 | 0.0017 |
| diabetes | pca_median | sensitivity | -0.1012 | 0.0103 |
| diabetes | pca_median | specificity | +0.0273 | 0.0076 |
| diabetes | pca_median | f1_score | -0.1151 | 0.0037 |
| diabetes | svd_median | outer_roc_auc | -0.0005 | 0.0029 |
| diabetes | svd_median | balanced_accuracy | +0.0073 | 0.0360 |
| diabetes | svd_median | sensitivity | +0.0020 | 0.1019 |
| diabetes | svd_median | specificity | +0.0127 | 0.0332 |
| diabetes | svd_median | f1_score | +0.0206 | 0.1284 |
| heart_disease | baseline_median | outer_roc_auc | -0.0011 | 0.0022 |
| heart_disease | baseline_median | balanced_accuracy | -0.0048 | 0.0056 |
| heart_disease | baseline_median | sensitivity | -0.0095 | 0.0149 |
| heart_disease | baseline_median | specificity | -0.0000 | 0.0062 |
| heart_disease | baseline_median | f1_score | -0.0063 | 0.0068 |
| heart_disease | mean_imputation | outer_roc_auc | -0.0011 | 0.0022 |
| heart_disease | mean_imputation | balanced_accuracy | -0.0048 | 0.0056 |
| heart_disease | mean_imputation | sensitivity | -0.0095 | 0.0149 |
| heart_disease | mean_imputation | specificity | -0.0000 | 0.0062 |
| heart_disease | mean_imputation | f1_score | -0.0063 | 0.0068 |
| heart_disease | pca_median | outer_roc_auc | -0.0002 | 0.0038 |
| heart_disease | pca_median | balanced_accuracy | -0.0320 | 0.0446 |
| heart_disease | pca_median | sensitivity | -0.0640 | 0.0995 |
| heart_disease | pca_median | specificity | +0.0000 | 0.0105 |
| heart_disease | pca_median | f1_score | -0.0658 | 0.1011 |
| heart_disease | svd_median | outer_roc_auc | +0.0032 | 0.0007 |
| heart_disease | svd_median | balanced_accuracy | -0.0018 | 0.0617 |
| heart_disease | svd_median | sensitivity | +0.0105 | 0.1417 |
| heart_disease | svd_median | specificity | -0.0141 | 0.0286 |
| heart_disease | svd_median | f1_score | +0.0013 | 0.1542 |

## Runtime

| Seed | Elapsed seconds | Elapsed hours |
|---:|---:|---:|
| 42 | 1096.17 | 0.30 |
| 43 | 2125.34 | 0.59 |
| 44 | 3230.38 | 0.90 |
| Total sequential elapsed | 6451.90 | 1.79 |

## Validation note

The first in-process validator used exact winner-evaluation ID matching. It
rejected seeds 43 and 44 where near-tied recorded losses differed only below
the CSV precision used by the runner. The validator was corrected to allow a
selected score within 1e-12 of the best recorded criterion; all three saved
seed runs were revalidated successfully. No candidate searches were rerun,
and seeds 43/44 were marked complete only after that revalidation.

## Limitations

- Outer folds overlap; the fold-level comparisons are not independent.
- The three seeds reuse the same underlying datasets; do not treat seed/fold rows as independent observations.
- These summaries are descriptive; no statistical significance is claimed.
- SVD is explicitly seeded per run. Historical v2 outputs were not modified or regenerated by this experiment.
- Runtime is specific to this execution environment.
