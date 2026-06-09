import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_validate
from bayes_model.sklearn_wrapper import BayesSklearnWrapper


def find_optimal_features(
    data: pd.DataFrame, target_column: str, n_splits: int = 25, test_size: float = 0.3
) -> dict:
    """
    Find optimal feature set by testing correlation thresholds.

    Uses cross-validation to test different correlation thresholds and returns
    the configuration with the best F1-score.

    Args:
        data: Dataset with features and target.
        target_column: Name of target column.
        n_splits: Number of shuffle splits for cross-validation.
        test_size: Proportion of test set in each split.

    Returns:
        Dictionary with 'f1', 'dropped', and 'threshold' keys.
    """
    # Prepare numerical data for correlation analysis
    numerical_data = data.select_dtypes(include=[np.number]).copy()
    if target_column in numerical_data.columns:
        numerical_data = numerical_data.drop(columns=[target_column])

    # Map categorical features to numeric for correlation
    if "BMI Category" in data.columns:
        bmi_mapping = {"Normal": 0, "Overweight": 1, "Obese": 2}
        numerical_data["BMI Category"] = data["BMI Category"].map(bmi_mapping)

    if "Gender" in data.columns:
        numerical_data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

    # Calculate Spearman correlation matrix
    corr_matrix = numerical_data.corr(method="spearman")
    corr_matrix = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Extract correlation thresholds to test
    thresholds = (
        corr_matrix.where(corr_matrix.abs() > 0.75)
        .stack()
        .dropna()
        .abs()
        .tolist()
    )
    thresholds.append(0.75)
    thresholds = list(set(thresholds))  # Remove duplicates

    # Cross-validation strategy
    cv_strategy = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    results = []

    # Test each correlation threshold
    for threshold in thresholds:
        # Find highly correlated features to drop
        mask = (corr_matrix.abs() > threshold).any(axis=1) & (
            ~corr_matrix.isna().all(axis=1)
        )
        columns_to_drop = corr_matrix[mask].index.tolist()

        # Prepare dataset
        data_temp = data.drop(columns=columns_to_drop, errors="ignore")
        X = data_temp.drop(columns=[target_column])
        y = data_temp[target_column]

        # Train and evaluate
        wrapper = BayesSklearnWrapper(target_column=target_column)
        cv_results = cross_validate(
            wrapper,
            X,
            y,
            cv=cv_strategy,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        results.append(
            {
                "f1": cv_results["test_score"].mean(),
                "dropped": columns_to_drop,
                "threshold": threshold,
            }
        )

    # Find best configuration
    results.sort(key=lambda x: x["f1"], reverse=True)
    best_result = results[0]

    print(f"Best F1-score: {best_result['f1']:.4f} for threshold {best_result['threshold']}")
    print(f"Dropped columns: {best_result['dropped']}")

    return best_result
