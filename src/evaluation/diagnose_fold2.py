import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data_loader import get_dataset
from src.evaluation.nested_cv_experiment import (
    RANDOM_STATE,
    OUTER_FOLDS,
    PREPROCESSING_CONFIGS,
    create_model,
)


def main():
    # 1. Load the same diabetes dataset as the main experiment.
    X, y, categorical_features = get_dataset("diabetes")

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    # 2. Recreate the original outer CV split exactly.
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(outer_cv.split(X, y))

    # Fold 2: Python indexing starts at zero, so index 1 means fold 2.
    train_indices, test_indices = splits[1]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("=== OUTER FOLD 2 DATA ===")
    print("Training labels:", y_train.value_counts().sort_index().to_dict())
    print("Test labels:", y_test.value_counts().sort_index().to_dict())

    # 3. Use the baseline preprocessing configuration.
    config = next(
        config
        for config in PREPROCESSING_CONFIGS
        if config["name"] == "baseline_median"
    )

    # 4. Compare the selected models with a reference configuration.
    candidates = {
        "Standard selected": {
            "C": 0.050919767891807195,
            "gamma": 0.2212689390626141,
        },
        "Uncertainty-aware selected": {
            "C": 0.001,
            "gamma": 0.009364872153361016,
        },
        "Diagnostic C=1": {
            "C": 1.0,
            "gamma": 0.1,
        },
    }

    # 5. Fit each model and compare training/test predictions.
    for method, candidate in candidates.items():
        model = create_model(
            C=candidate["C"],
            gamma=candidate["gamma"],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_imputation=config["numeric_imputation"],
            dimensionality_reduction=config["dimensionality_reduction"],
        )

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        test_scores = model.decision_function(X_test)

        print(f"\n=== {method} ===")
        print(f"C: {candidate['C']}")
        print(f"Gamma: {candidate['gamma']}")
        print("SVM classes:", model.named_steps["svm"].classes_)

        print(
            "Training predictions:",
            pd.Series(train_predictions).value_counts().sort_index().to_dict(),
        )

        print(
            "Test predictions:",
            pd.Series(test_predictions).value_counts().sort_index().to_dict(),
        )

        print(f"Test decision score min: {test_scores.min():.6f}")
        print(f"Test decision score max: {test_scores.max():.6f}")


if __name__ == "__main__":
    main()