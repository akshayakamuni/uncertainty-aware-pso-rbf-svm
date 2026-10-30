# v2 Validation Report

| Check | Result | Details |
|---|---|---|
| Required results exist | PASS | All required files present. |
| All primary dataset/config/method/fold combinations represented | PASS | Found 240 primary rows; expected 240. |
| All separate calibrated combinations represented | PASS | Found 240 calibrated rows; expected 240. |
| No duplicate result keys | PASS | Checked default and calibrated result keys. |
| Candidate searches have documented 48-evaluation budgets | PASS | Rows=17280; expected 17,280; each of 360 searches has 48 rows. |
| No duplicate candidate evaluation keys | PASS | Checked dataset/config/fold/search/evaluation identifiers. |
| All result metrics and runtimes finite | PASS | Checked default, calibrated, and lambda sensitivity rows. |
| Metrics and hyperparameters within valid ranges | PASS | AUC/class metrics/losses in [0,1], SD/runtime nonnegative, C/gamma in bounds. |
| Selected candidates belong to their recorded pools | PASS | Evaluation IDs and candidate parameters/loss statistics matched. |
| Final selection criteria recompute correctly | PASS | Mean loss for M1/M4; mean + lambda*sample SD for M2/M3. |
| M1 and M2 use an identical candidate pool | PASS | Same M1 history ID and all 48 evaluations per matched fold. |
| Fitness uses sample SD and recorded fold losses | PASS | Recomputed means, ddof=1 SDs, and lambda-penalized fitness. |
| M3 personal/global bests use penalized fitness | PASS | Replayed logged best-so-far traces and verified two update_best calls in PSO loop. |
| M3 lambda sensitivity reruns the penalized search | PASS | All 5 declared lambdas have 60 fold/config rows and 48-evaluation pools. |
| Matched initial PSO positions across objectives | PASS | Matched seed generated equal particle start coordinates per fold. |
| Current runner controls stochastic operation seeds | PASS | Current source seeds outer/inner stratified splits, both search RNGs, and TruncatedSVD with seed 42. Historical config retains svd_random_state=None; stored outputs remain from that original run and were not regenerated. |
| M4 is a separate equal-budget random-search control | PASS | Random pool has 48 in-bound candidates per fold and no PSO-best updates. |
| Tuning and thresholds exclude outer-test data | PASS | Checked runner calls use only X_train/y_train for search and inner OOF thresholds. |
| Default-threshold and calibrated outputs remain separate | PASS | Separate result files and explicit threshold-scope fields. |
| Search runtimes and evaluation counts recorded | PASS | Runtime rows=360; expected 360 searches. |
| All dataset/configuration runs completed | PASS | 12/12 completed. |
| Output directory isolated from prior experiment folders | PASS | C:\Users\saira\Downloads\pso_final_extracted\uncertainty-aware-pso-svm\experiments\uncertainty_aware_pso_v2 |
| No incomplete temporary CSV files remain | PASS | None. |

The best-update validation independently replays the persisted per-particle and global best fitness traces and checks the runner calls the shared update function in its PSO loop. The companion `test_experiment.py` provides a deterministic synthetic test in which the mean-loss and penalized-fitness choices differ.

The SVD seed check validates the current runner only. The recorded experiment config and result files retain their original `svd_random_state: null`; these historical results were not regenerated or independently reproduced.
