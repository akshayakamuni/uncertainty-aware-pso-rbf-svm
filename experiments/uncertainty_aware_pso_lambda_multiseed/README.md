# Multi-seed lambda sensitivity experiment

This is a new, isolated study comparing standard PSO (lambda=0) with
uncertainty-aware PSO at lambda 0.25, 0.5, 1, and 2. It imports the completed
v2 runner read-only for model construction, candidate evaluation, PSO movement,
fitness, best updates, selection, and scoring. It does not modify v1, v2, or
the prior M1/M3 multi-seed experiment.

## Protocol

- Seeds 42, 43, and 44.
- Three established datasets and four established preprocessing configurations.
- Stratified shuffled five-fold outer and five-fold inner CV.
- Eight particles × six iterations = 48 candidate evaluations per
  dataset/configuration/fold/lambda search.
- Search bounds remain `log10(C), log10(gamma) ∈ [-3, 3]`.
- Lambda 0 uses the exact v2 PSO function with mean inner-fold loss as fitness;
  personal-best/global-best updates and final selection therefore use the same
  mean-loss criterion as M1. Nonzero lambda runs use a separate PSO trajectory
  and criterion `mean loss + lambda × sample SD` during fitness, both best
  updates, and final candidate selection. This is not a post-hoc-only change.
- All five lambda searches share the same outer and inner split objects for a
  seed/configuration/fold. Each search restarts the seeded PSO RNG; recorded
  initial particle coordinates are compared explicitly before scoring.
- Outer folds use `seed`; inner folds and PSO use `seed + outer_fold`.
  TruncatedSVD is explicitly seeded with the per-run seed. The seed values are
  written to each effective run config and result row.
- Preprocessing is built and fitted inside each CV training fold. Candidate
  tuning and OOF threshold selection use outer-training data only. Default and
  threshold-calibrated metrics are persisted separately.
- A seed is marked complete only after its seed validator passes. Failed
  folders preserve logs and partial outputs. Existing seed folders or
  aggregate files cause an immediate refusal to overwrite.

## Commands

After approval, run the full experiment:

```powershell
python experiments\uncertainty_aware_pso_lambda_multiseed\run_lambda_multiseed.py --seeds 42 43 44
```

The command runs all five lambda searches for each seed and creates outputs
only in `seed_42`, `seed_43`, and `seed_44` under this directory. The runner
creates paired aggregates only if every configured seed validates.

Validate the static design without running the experiment, a single output
seed, or all results:

```powershell
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py --check-config
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py --seed 42
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py
```

Recover a complete saved seed after a validator correction without rerunning
candidate searches:

```powershell
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py --seed 42 --promote-seed
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py --aggregate
```

Lightweight tests and compilation:

```powershell
python -m unittest discover -s experiments\uncertainty_aware_pso_lambda_multiseed -p test_lambda_multiseed.py -v
python -m py_compile experiments\uncertainty_aware_pso_lambda_multiseed\run_lambda_multiseed.py experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py experiments\uncertainty_aware_pso_lambda_multiseed\test_lambda_multiseed.py
python experiments\uncertainty_aware_pso_lambda_multiseed\validate_lambda_multiseed.py --check-config
```

## Outputs

Each seed directory contains an effective config, default-threshold outer-fold
results, separate threshold-calibrated outer-fold results, all 48-candidate
histories, per-search runtime, completion/failure logs, and a validation
report. Root-level paired files compare each nonzero lambda with lambda=0:

- `paired_lambda_fold_differences.csv`
- `paired_lambda_calibrated_fold_differences.csv`
- `paired_lambda_summary_by_seed.csv`
- `paired_lambda_calibrated_summary_by_seed.csv`
- `paired_lambda_summary_across_seeds.csv`
- `paired_lambda_calibrated_summary_across_seeds.csv`
- `findings_summary.md`

Comparisons are paired by seed, dataset, preprocessing configuration, and
outer fold. Outer folds overlap, and seeds reuse the same data; do not treat
these rows as independent or infer statistical significance from them.

## Computational budget and runtime estimate

Per seed: `3 datasets × 4 configurations × 5 outer folds × 5 lambdas × 48`
= **14,400 candidate evaluations** across **300 searches**.

Three seeds: **43,200 candidate evaluations** across **900 searches**, plus
final model fitting/scoring and inner OOF threshold calibration.

In the recorded v2 `runtime_summary.csv`, the five PSO lambda searches took
about 34,342.64 seconds for one seed (the 0.25 and 0.5 totals were unusually
large). A linear three-seed estimate is about **28.6 hours of search time**,
before validation and calibration overhead. The prior multi-seed M1/M3 run
took 1.79 hours for two lambda trajectories across three seeds, suggesting
substantial machine/run variation. This is an estimate, not a wall-clock
guarantee.

## Findings summary status

`findings_summary.md` is a pre-run placeholder until all configured seeds have
passed validation and aggregate outputs are generated.
