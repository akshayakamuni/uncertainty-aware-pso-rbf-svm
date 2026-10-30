
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features,
    categorical_features=None,
    numeric_imputation="median",
    dimensionality_reduction=None,
    pca_variance=0.95,
    svd_components=10,
):
    """
    Build a leakage-safe preprocessing pipeline.

    Numeric features:
        1. Impute missing values using mean or median.
        2. Standardize the values.

    Categorical features:
        1. Impute using the most frequent category.
        2. One-hot encode categories.

    Optional dimensionality reduction:
        None: original feature representation.
        "pca": retain the requested proportion of variance.
        "svd": reduce to a fixed number of components.

    The returned pipeline must be fitted only on training data.
    """

    if categorical_features is None:
        categorical_features = []

    if numeric_imputation not in ("mean", "median"):
        raise ValueError(
            "numeric_imputation must be 'mean' or 'median'."
        )

    if dimensionality_reduction not in (None, "pca", "svd"):
        raise ValueError(
            "dimensionality_reduction must be None, 'pca', or 'svd'."
        )

    if not 0 < pca_variance <= 1:
        raise ValueError("pca_variance must be in (0, 1].")

    if svd_components < 1:
        raise ValueError("svd_components must be at least 1.")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=numeric_imputation),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, numeric_features)
    ]

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    steps = [("columns", column_transformer)]

    if dimensionality_reduction == "pca":
        steps.append(
            ("reduction", PCA(n_components=pca_variance))
        )
    elif dimensionality_reduction == "svd":
        steps.append(
            ("reduction", TruncatedSVD(n_components=svd_components))
        )

    return Pipeline(steps=steps)
