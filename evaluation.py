import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from model import create_preprocessor, create_pipeline, get_feature_importances


def analyze_correlations(df: pd.DataFrame) -> None:
    """Plots a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.drop("Person ID", axis=1, errors="ignore")

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def print_feature_importances(
    all_feature_names: list,
    importances: np.ndarray,
    indices: np.ndarray,
    title: str = "Top 5 Feature Importances",
) -> None:
    """Prints top most important features."""
    print(f"\n{title}:")

    # iterate over the top 8 sorting indices
    for f in range(min(8, len(all_feature_names))):
        print(f"{all_feature_names[indices[f]]}: {importances[indices[f]]:.4f}")


def train_and_evaluate(df: pd.DataFrame, binary: bool = False) -> None:
    """Evaluates model using 5-Fold Cross Validation, prints detailed report and extracts importances."""
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    if binary:
        # map target to 0 for healthy and 1 for any disorder
        y = df["Sleep Disorder"].apply(lambda x: 0 if x == "No Disorder" else 1)
        feature_title = "Top 8 Feature Importances (Binary Classification)"
        print("\nStarting 5-Fold Cross-Validation for Binary Classification...")
        display_labels = ["No Disorder", "Disorder"]
    else:
        y = df["Sleep Disorder"]
        feature_title = "Top 8 Feature Importances (Multiclass Classification)"
        print("\nStarting 5-Fold Cross-Validation for Multiclass Classification...")
        display_labels = None

    preprocessor = create_preprocessor()
    pipeline = create_pipeline(preprocessor)

    # configure k-fold split maintaining class distributions
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # define scoring metrics
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    # execute validation process automatically over 5 iterations
    cv_results = cross_validate(pipeline, X, y, cv=cv_strategy, scoring=scoring)

    # calculate and output the mean performance across all validation folds
    print("\nAverage Scores from 5-Fold Cross-Validation:")
    print(f"Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(cv_results['test_precision_weighted']):.4f}")
    print(f"Recall:    {np.mean(cv_results['test_recall_weighted']):.4f}")
    print(f"F1-score:  {np.mean(cv_results['test_f1_weighted']):.4f}")

    # generates aggregated predictions from all 5 folds to build a single matrix and report
    y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv_strategy)

    # generate detailed classification report for latex tables
    print("\nDetailed Classification Report (CV):")
    if display_labels:
        print(classification_report(y, y_pred_cv, target_names=display_labels))
    else:
        print(classification_report(y, y_pred_cv))

    # creates and displays the confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred_cv,
        cmap="Blues",
        display_labels=display_labels,
        xticks_rotation=0 if binary else 45,
    )
    matrix_title = "Binary Confusion Matrix (CV)" if binary else "Confusion Matrix (CV)"
    plt.title(matrix_title)
    plt.tight_layout()
    plt.show()

    # fit the model on the entire dataset to maximize rule extraction quality
    pipeline.fit(X, y)

    # retrieve and display final metric weights
    all_feature_names, importances, indices = get_feature_importances(pipeline)
    print_feature_importances(
        all_feature_names, importances, indices, title=feature_title
    )
