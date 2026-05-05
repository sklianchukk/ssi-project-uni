import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Creates a column transformer for categorical and numeric features."""
    # define categorical transformation logic
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        # leave numeric columns unchanged
        remainder="passthrough",
    )


def create_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Creates a pipeline with preprocessor and random forest classifier."""
    # construct pipeline to prevent data leakage during cross-validation
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )


def get_feature_importances(pipeline: Pipeline) -> tuple:
    """Extracts feature names and importance scores from trained pipeline."""
    # access the underlying random forest model
    rf_model = pipeline.named_steps["classifier"]

    # extract dynamically generated column names from the encoder
    cat_feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )

    all_feature_names = list(cat_feature_names) + NUMERIC_FEATURES
    importances = rf_model.feature_importances_

    # sort indices based on importance scores in descending order
    indices = np.argsort(importances)[::-1]

    return all_feature_names, importances, indices
