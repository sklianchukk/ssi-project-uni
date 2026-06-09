import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from common_utils.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Create column transformer for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )


def create_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create pipeline with preprocessor and Random Forest classifier."""
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
    """Extract feature names and importance scores from trained pipeline."""
    rf_model = pipeline.named_steps["classifier"]

    # Extract feature names from encoder
    cat_feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )

    all_feature_names = list(cat_feature_names) + NUMERIC_FEATURES
    importances = rf_model.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    return all_feature_names, importances, indices
