"""
Training and evaluation script for Random Forest model.

Uses GridSearchCV to find optimal hyperparameters and evaluates with stratified k-fold cross-validation.
Generates feature importance plots and confusion matrices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)

from common_utils.data_processing import load_dataset, preprocess_data_for_random_forest
from common_utils.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from common_utils.evaluation_utils import (
    ensure_output_folder,
    plot_class_distribution,
    plot_correlation_matrix,
    print_metrics,
    plot_confusion_matrix,
    print_feature_importances,
    plot_feature_importances,
    print_gridsearch_results,
)
from random_forest_model.pipeline import create_preprocessor, create_pipeline, get_feature_importances


def train_and_evaluate(df: pd.DataFrame, binary: bool = False, folder: str = None) -> None:
    """
    Train and evaluate Random Forest with GridSearchCV and cross-validation.

    Args:
        df: Input dataset.
        binary: If True, converts to binary classification (Disorder vs No Disorder).
        folder: Folder to save plots and visualizations.
    """
    # Prepare data
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    if binary:
        y = df[TARGET_COLUMN].apply(lambda x: 0 if x == "No Disorder" else 1)
        title = "Binary Classification (Disorder vs No Disorder)"
        display_labels = ["No Disorder", "Disorder"]
        plot_filename = "confusion_matrix_rf_binary.png"
        importance_filename = "feature_importance_rf_binary.png"
    else:
        y = df[TARGET_COLUMN]
        title = "Multiclass Classification"
        display_labels = None
        plot_filename = "confusion_matrix_rf_multiclass.png"
        importance_filename = "feature_importance_rf_multiclass.png"

    print(f"\n{'=' * 60}")
    print(f"RANDOM FOREST - {title}")
    print('=' * 60)

    # Setup pipeline and cross-validation
    preprocessor = create_preprocessor()
    pipeline = create_pipeline(preprocessor)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search for optimal hyperparameters
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5],
    }

    print("\nExecuting GridSearch Optimization...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    print_gridsearch_results(grid_search, top_n=5)
    best_pipeline = grid_search.best_estimator_

    # Cross-validation evaluation
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv_results = cross_validate(best_pipeline, X, y, cv=cv_strategy, scoring=scoring)

    # Make predictions
    y_pred = cross_val_predict(best_pipeline, X, y, cv=cv_strategy)

    # Print metrics
    print_metrics(y, y_pred, cv_results=cv_results, label="5-Fold Cross-Validation Results")

    # Plot confusion matrix
    plot_confusion_matrix(
        y,
        y_pred,
        filename=plot_filename,
        title=f"Confusion Matrix - {title}",
        display_labels=display_labels,
        rotation=0 if binary else 45,
        folder=folder,
    )

    # Fit on full data for feature importances
    best_pipeline.fit(X, y)
    all_features, importances, indices = get_feature_importances(best_pipeline)

    print_feature_importances(
        all_features,
        importances,
        indices,
        title=f"Top 8 Feature Importances - {title}",
    )

    plot_feature_importances(
        all_features,
        importances,
        indices,
        filename=importance_filename,
        title=f"Top Feature Importances - {title}",
        folder=folder,
    )


def main():
    """Main training pipeline."""
    # Create output folder
    output_folder = ensure_output_folder("Evaluation_images")

    # Load and preprocess data
    print("Loading dataset...")
    df_raw = load_dataset("sleep_quality.csv")
    df_processed = preprocess_data_for_random_forest(df_raw)

    # Save processed data
    df_processed.to_csv("sleep_processed_rf.csv", index=False)
    print(f"Processed dataset saved: {len(df_processed)} samples")

    # Exploratory visualizations
    plot_class_distribution(df_processed, folder=output_folder)
    plot_correlation_matrix(df_processed, exclude_cols=["Sleep Disorder"], folder=output_folder)

    # Multiclass classification
    train_and_evaluate(df_processed, binary=False, folder=output_folder)

    # Binary classification
    train_and_evaluate(df_processed, binary=True, folder=output_folder)

    plt.show()


if __name__ == "__main__":
    main()
