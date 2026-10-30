
from sklearn.model_selection import train_test_split

from src.data_loader import get_dataset
from src.evaluation.nested_cv_experiment import create_model


def main():
    X, y, categorical_features = get_dataset("diabetes")

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    configurations = [
        ("Baseline", "median", None),
        ("PCA", "median", "pca"),
        ("SVD", "median", "svd"),
        ("Mean imputation", "mean", None),
    ]

    for name, imputation, reduction in configurations:
        print(f"\nTesting: {name}")

        model = create_model(
            C=1.0,
            gamma=0.1,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Replace the preprocessing step with the requested configuration.
        from src.preprocessing import build_preprocessor

        model.set_params(
            preprocessing=build_preprocessor(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_imputation=imputation,
                dimensionality_reduction=reduction,
                pca_variance=0.95,
                svd_components=5,
            )
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("Training samples:", len(X_train))
        print("Test samples:", len(X_test))
        print("Predictions produced:", len(predictions))
        print("Accuracy (sanity check only):", model.score(X_test, y_test))
        print("Pipeline fit and prediction: OK")


if __name__ == "__main__":
    main()
