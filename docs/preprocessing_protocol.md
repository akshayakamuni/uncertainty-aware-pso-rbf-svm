
# Preprocessing Protocol

## 1. Breast Cancer Dataset

- Use the 30 numerical diagnostic features.
- Exclude the sample ID from model inputs.
- Encode malignant as 1 and benign as 0.
- Apply median imputation and standard scaling inside the model pipeline.

## 2. Heart Disease Dataset

- Use 13 input features.
- Convert the original target into binary labels:
  - 0 = no disease
  - 1–4 = disease present
- Treat the selected coded categorical features using most-frequent imputation and one-hot encoding.
- Treat the remaining features as numerical and apply median imputation and standard scaling.
- Handle missing values in `ca` and `thal` inside the model pipeline.

## 3. Diabetes Dataset

- Use eight input features.
- Preserve pregnancies equal to zero.
- Treat zero values in glucose, blood pressure, skin thickness, insulin, and BMI as potentially missing.
- Replace those selected zeros with missing values.
- Apply median imputation and standard scaling inside the model pipeline.
- Encode the target as 0 or 1.

## 4. Leakage Prevention

- Fit imputers, scalers, and encoders only on each training fold.
- Keep preprocessing inside the scikit-learn pipeline.
- Use the same evaluation splits when comparing model-selection methods.

## 5. Important Limitation

The interpretation of selected diabetes zero values as missing measurements is a working assumption. Its effect should be assessed through a sensitivity analysis.
