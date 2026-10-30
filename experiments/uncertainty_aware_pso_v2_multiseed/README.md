# Multi-seed M1 vs M3 robustness experiment

This isolated runner compares only **M1 Standard PSO** and **M3 UA-PSO**
(`lambda=1`). It reuses the v2 implementation read-only for preprocessing,
fitness, PSO movement, candidate recording, selection, and outer-test scoring.
It does not alter the original or v2 runner.

## Protocol

- Seeds: 42, 43, and 44.
- Datasets: diabetes, breast cancer, and heart disease.
- Preprocessing: baseline median, mean imputation, PCA median, and SVD median.
- Nested CV: shuffled stratified five-fold outer and inner splits.
- Search: eight particles, six iterations, 48 evaluations per method/search,
  with `log10(C)` and `log10(gamma)` in `[-3, 3]`.
- M1 selects the minimum mean inner-fold loss.
- M3 uses mean inner-fold loss plus one times the sample standard deviation
  during PSO fitness/best updates and final selection.
- Outer folds use the run seed; inner folds use `seed + outer_fold`; both PSO
  searches restart the same seeded generator for each matched outer fold.
  The runner compares initial coordinates directly and fails the fold if they
  differ.
- Every SVD pipeline sets `TruncatedSVD.random_state` to that run's seed. Each
  per-seed config records this actual value. The historical v2 configuration
  remains unchanged.
- Each seed is validated before its completion log is marked completed.
  Existing `seed_<n>` directories cause an immediate refusal to overwrite.

## Commands

Run all configured seeds:

```powershell
python experiments\uncertainty_aware_pso_v2_multiseed\run_multiseed.py
```

Or run a subset of configured seeds into still-absent directories:

```powershell
python experiments\uncertainty_aware_pso_v2_multiseed\run_multiseed.py --seeds 42
python experiments\uncertainty_aware_pso_v2_multiseed\run_multiseed.py --seeds 43 44
```

Each seed is saved independently under `seed_42`, `seed_43`, or `seed_44`.
The runner builds cross-seed paired tables only after all three validated runs
are present. It never resumes or replaces a pre-existing seed directory.

Validate the design without experiment outputs, or validate outputs after a
run:

```powershell
python experiments\uncertainty_aware_pso_v2_multiseed\validate_multiseed.py --check-config
python experiments\uncertainty_aware_pso_v2_multiseed\validate_multiseed.py --seed 42
python experiments\uncertainty_aware_pso_v2_multiseed\validate_multiseed.py
```

Run lightweight tests and syntax compilation:

```powershell
python -m unittest discover -s experiments\uncertainty_aware_pso_v2_multiseed -p test_multiseed.py -v
python -m py_compile experiments\uncertainty_aware_pso_v2_multiseed\run_multiseed.py experiments\uncertainty_aware_pso_v2_multiseed\validate_multiseed.py experiments\uncertainty_aware_pso_v2_multiseed\test_multiseed.py
```

## Outputs and interpretation

Each seed directory contains an effective config, outer-fold results, all
candidate histories, runtimes, completion/failure logs, and a validation
report. The experiment root receives:

- `paired_fold_differences.csv`: M3-minus-M1 metric differences paired by seed,
  dataset, preprocessing configuration, and outer fold.
- `paired_summary_by_seed.csv`: descriptive fold-level mean and SD within each
  seed/dataset/configuration.
- `paired_summary_across_seeds.csv`: descriptive variation in seed-level
  means for each dataset/configuration.
- `aggregation_notes.md`: dependence and inference limitations.

Outer CV folds overlap, and seeds use the same underlying datasets. Neither
folds nor seed/fold combinations are independent statistical samples. These
summaries are descriptive; no significance test is claimed.

## Expected workload

Each seed includes 60 matched dataset/configuration/fold searches per method:
2,880 evaluations per method and 5,760 combined. Three seeds require 17,280
candidate evaluations, plus final model fitting/scoring. Runtime will vary by
machine and dataset.
