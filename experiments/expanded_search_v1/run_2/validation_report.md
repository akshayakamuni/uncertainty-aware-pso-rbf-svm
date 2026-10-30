# Validation Report

| Check | Result | Details |
|---|---|---|
| Required output files exist | PASS | All required files present. |
| All dataset/configuration folds and methods represented | PASS | default rows=120, calibrated rows=120, combinations=12/12 |
| All candidate evaluations recorded | PASS | 2880/2880 rows |
| No duplicate fold-result keys | PASS | Checked default and calibrated fold result keys. |
| No duplicate candidate evaluation keys | PASS | Checked dataset/configuration/fold/evaluation keys. |
| Default metrics finite and in valid ranges | PASS | Validated 120 rows. |
| Calibrated metrics finite and in valid ranges | PASS | Validated 120 rows. |
| Selected candidate belongs to its recorded pool | PASS | Candidate evaluation ID, C, gamma, mean and sample SD matched. |
| Selection criterion is computed correctly | PASS | Recomputed mean loss (standard) and mean + lambda*sample SD. |
| Both selectors use the same recorded candidate pool | PASS | Per-fold evaluation count and selected pool size agree. |
| Tuning and thresholds exclude outer-test data | PASS | Runner passes outer-training data to search/inner OOF calibration; metadata flags were checked. |
| Output paths isolated to designated directory | PASS | C:\Users\saira\Downloads\pso_final_extracted\uncertainty-aware-pso-svm\experiments\expanded_search_v1\run_2 |
| Baseline comparison uses compatible evaluation conventions | PASS | Matched fold/method rows=120; baseline and expanded metrics use default threshold and same configured outer seed. |
| Baseline delta summary is complete and finite | PASS | Expected three datasets by two selection methods. |
| All 12 configurations completed | PASS | 12/12 completed. |

Code-level leakage check: the PSO function accepts only the outer training partition; outer-test indices are passed only to final metric evaluation. Calibrated thresholds are selected from inner-fold out-of-fold scores generated on outer-training data.
