# v2 Findings Summary

M3 uses mean inner-fold loss plus lambda times sample SD during particle fitness and best updates. M2 changes only final selection from M1's shared candidate pool.

## Primary default-threshold results by dataset

Means pool four preprocessing variants and five outer folds and are descriptive only.

| Dataset | Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | M1 Standard PSO | 0.9945 | 0.9653 | 0.9447 | 0.9860 | 0.9595 | 0.0018 |
| breast_cancer | M2 Post-hoc UA | 0.9950 | 0.9665 | 0.9470 | 0.9860 | 0.9610 | 0.0010 |
| breast_cancer | M3 UA-PSO | 0.9946 | 0.9675 | 0.9505 | 0.9846 | 0.9616 | 0.0014 |
| breast_cancer | M4 Random | 0.9954 | 0.9618 | 0.9389 | 0.9846 | 0.9550 | 0.0003 |
| diabetes | M1 Standard PSO | 0.8333 | 0.6972 | 0.5144 | 0.8800 | 0.5777 | 0.0010 |
| diabetes | M2 Post-hoc UA | 0.8342 | 0.6921 | 0.4982 | 0.8860 | 0.5613 | -0.0016 |
| diabetes | M3 UA-PSO | 0.8347 | 0.6849 | 0.4699 | 0.9000 | 0.5301 | -0.0033 |
| diabetes | M4 Random | 0.8291 | 0.6710 | 0.4360 | 0.9060 | 0.5059 | 0.0030 |
| heart_disease | M1 Standard PSO | 0.9049 | 0.8162 | 0.7372 | 0.8952 | 0.7718 | -0.0019 |
| heart_disease | M2 Post-hoc UA | 0.9046 | 0.8113 | 0.7319 | 0.8906 | 0.7665 | -0.0024 |
| heart_disease | M3 UA-PSO | 0.9075 | 0.8332 | 0.7818 | 0.8845 | 0.8132 | -0.0035 |
| heart_disease | M4 Random | 0.9071 | 0.7999 | 0.7106 | 0.8892 | 0.7326 | -0.0049 |

## Overall descriptive comparison

Unweighted means over the 60 dataset/configuration/fold combinations per method; these observations are overlapping and not independent.

| Method | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | Loss gap |
|---|---:|---:|---:|---:|---:|---:|
| M1 Standard PSO | 0.9109 | 0.8262 | 0.7321 | 0.9204 | 0.7697 | 0.0003 |
| M2 Post-hoc UA | 0.9113 | 0.8233 | 0.7257 | 0.9209 | 0.7629 | -0.0010 |
| M3 UA-PSO | 0.9123 | 0.8286 | 0.7341 | 0.9230 | 0.7683 | -0.0018 |
| M4 Random | 0.9105 | 0.8109 | 0.6952 | 0.9266 | 0.7312 | -0.0005 |

## Paired M3 minus M1 differences

Mean paired fold/configuration differences; no inferential test is performed.

| Dataset | Δ ROC-AUC | Δ balanced accuracy | Δ sensitivity | Δ specificity | Δ F1 | Δ loss gap |
|---|---:|---:|---:|---:|---:|---:|
| diabetes | 0.0014 | -0.0123 | -0.0445 | 0.0200 | -0.0476 | -0.0042 |
| breast_cancer | 0.0001 | 0.0022 | 0.0058 | -0.0014 | 0.0022 | -0.0003 |
| heart_disease | 0.0026 | 0.0170 | 0.0446 | -0.0107 | 0.0414 | -0.0015 |

## Hypothesis-level observations

- **H1 (trajectory/selection change):** M1 and M3 had different post-initial particle positions in all 60 matched searches; their selected C/gamma differed in 51 of 60 outer-fold comparisons.
- **H2 (predictive improvement):** M3 minus M1 mean ROC-AUC was +0.0014, balanced accuracy +0.0023, and F1 -0.0013 across the matched pooled rows. Effects varied by dataset.
- **H3 (seed stability):** not tested; only one seed was run.
- **H4 (generalization gap):** M3's signed mean gap was -0.0018 versus +0.0003 for M1, but a negative gap is not automatically better. The absolute gap was smaller for M3 in only 24/60 matched rows; no consistent gap reduction is established.

## M3 lambda sensitivity

Each nonzero lambda has its own PSO trajectory; lambda 0 reuses M1. Means pool four configurations and five folds per dataset.

| Dataset | Lambda | ROC-AUC | Balanced accuracy | F1 | Mean inner loss | Mean inner SD |
|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | 0 | 0.9945 | 0.9653 | 0.9595 | 0.0037 | 0.0039 |
| breast_cancer | 0.25 | 0.9948 | 0.9665 | 0.9608 | 0.0039 | 0.0035 |
| breast_cancer | 0.5 | 0.9950 | 0.9674 | 0.9621 | 0.0039 | 0.0033 |
| breast_cancer | 1 | 0.9946 | 0.9675 | 0.9616 | 0.0039 | 0.0032 |
| breast_cancer | 2 | 0.9952 | 0.9684 | 0.9633 | 0.0043 | 0.0031 |
| diabetes | 0 | 0.8333 | 0.6972 | 0.5777 | 0.1657 | 0.0347 |
| diabetes | 0.25 | 0.8347 | 0.7037 | 0.5888 | 0.1660 | 0.0335 |
| diabetes | 0.5 | 0.8350 | 0.6806 | 0.5249 | 0.1669 | 0.0317 |
| diabetes | 1 | 0.8347 | 0.6849 | 0.5301 | 0.1686 | 0.0300 |
| diabetes | 2 | 0.8347 | 0.6757 | 0.5054 | 0.1697 | 0.0295 |
| heart_disease | 0 | 0.9049 | 0.8162 | 0.7718 | 0.0970 | 0.0358 |
| heart_disease | 0.25 | 0.9044 | 0.8128 | 0.7681 | 0.0962 | 0.0335 |
| heart_disease | 0.5 | 0.9053 | 0.8129 | 0.7683 | 0.0959 | 0.0338 |
| heart_disease | 1 | 0.9075 | 0.8332 | 0.8132 | 0.0960 | 0.0317 |
| heart_disease | 2 | 0.9039 | 0.8032 | 0.7501 | 0.0973 | 0.0303 |

## Status and interpretation

- Completed configurations: 12/12; failures: 0.
- Total measured elapsed time across configurations: 37695.81 seconds (10.47 hours). This includes unusually long pauses recorded within some folds and is not a compute-only estimate.
- One seed was used. No repeated-seed stability conclusion is supported.
- The existing `TruncatedSVD` preprocessor uses `random_state=None`; outer/inner folds and PSO/random-search generators use seed 42 (with fold offsets), but the SVD solver itself was not seeded. Repeatability for SVD runs is therefore not guaranteed.
- Outer folds overlap and are not independent; no statistical significance is claimed.
- This three-dataset/configuration pooled table is descriptive, not a pooled clinical performance estimate.
- Threshold-calibrated results are separate; calibration uses outer-training inner OOF scores and inherits the score-scale mismatch when applied after refitting on all outer-training rows.
- Better accuracy from a larger search is not assumed; assess each method and dataset separately.
