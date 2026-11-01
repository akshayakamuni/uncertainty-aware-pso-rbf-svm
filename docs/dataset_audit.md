# Dataset Audit and Preprocessing Decisions

## 1. Wisconsin Diagnostic Breast Cancer

- Local file: `data/raw/breast_cancer/wdbc.data`.
- 569 samples; 32 source columns (ID, diagnosis, 30 numeric predictors).
- Class counts: 357 benign and 212 malignant.
- Missing values: none detected; duplicate rows: none detected.
- Model target: malignant = 1, benign = 0. The ID column is excluded.
- Predictors are standardized within each training fold.
- Dataset page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.

## 2. Cleveland Heart Disease

- Local file: `data/raw/heart_disease/processed.cleveland.data`.
- 303 samples; 13 predictors and one target.
- Original target counts: 0 = 164, 1 = 55, 2 = 36, 3 = 35, 4 = 13.
- Six missing cells were found in `ca` (4) and `thal` (2), represented as `?` and parsed as missing.
- Target mapping: original 0 -> no disease (0); original 1–4 -> disease present (1).
- `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, and `thal` are treated as categorical in the current loader; other features are numeric.
- Missing values are imputed inside the pipeline fitted on each training fold.
- Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease.

## 3. Pima Indians Diabetes

- Local file: `data/raw/diabetes/Diabetes.csv`.
- 768 samples; 8 predictors and one binary target. Class counts: 500 negative and 268 positive.
- No explicit NA values were found before preprocessing.
- Zero counts: glucose = 5, blood pressure = 35, skin thickness = 227, insulin = 374, BMI = 11.
- The loader treats zero in these five measurements as missing; zero pregnancies is retained as valid. This is a modeling assumption based on common handling of this dataset and should be stated as a limitation because not every zero can be proven to mean missing.
- The schema matches the commonly used Pima Indians Diabetes dataset. Exact provenance of this specific local file copy and its license remain unverified;
- Dataset description / original citation context: https://search.r-project.org/CRAN/refmans/reglogit/html/pima.html.

## 4. Reproducibility safeguards

- Raw datasets are excluded from Git tracking by `.gitignore`.
- Imputation, scaling, encoding, PCA, and SVD are fitted inside the pipeline within each training fold.
- Outer test folds are not used for candidate selection or threshold selection.
- Primary results use identical outer splits for Standard and Uncertainty-aware methods.
- The threshold-calibrated experiment selects its threshold using inner out-of-fold scores from the outer-training partition.
- See `evaluation_protocol.md` for implementation details and `references.md` for citations.

## 5. Remaining provenance task

Find and record the exact source URL/version for `Diabetes.csv` and verify redistribution/license terms for all local data files. Keep SHA-256 hashes in the release package so the experiment can be reproduced against the same file versions.
