"""
Training and evaluation script for Naive Bayes models.

Tests both KDE-based and Gaussian-based Naive Bayes classifiers on multiclass
and binary classification tasks with stratified k-fold cross-validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from common_utils.data_processing import load_dataset, preprocess_data_basic
from common_utils.evaluation_utils import plot_class_distribution, plot_correlation_matrix
from bayes_model.sklearn_wrapper import BayesSklearnWrapper
from bayes_model.classifier import BayesClassifier, BayesGaussianClassifier


def evaluate_classifier(
    dataset: pd.DataFrame,
    target_column: str,
    classifier,
    cv_splits: int = 5,
    title: str = "Classification Report",
) -> None:
    """
    Train and evaluate classifier using stratified k-fold cross-validation.

    Args:
        dataset: Training dataset.
        target_column: Name of target column.
        classifier: Classifier instance.
        cv_splits: Number of cross-validation splits.
        title: Title for the evaluation output.
    """
    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)

    # Prepare data
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    classes = sorted(y.unique().tolist())

    # Wrap classifier
    wrapper = BayesSklearnWrapper(target_column=target_column)
    wrapper.model = classifier

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(wrapper, X, y, cv=cv, n_jobs=-1)

    # Metrics
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    # Print per-class metrics
    print(f"\n{'Class':<20}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}")
    print("-" * 65)
    for cls in classes:
        p = report[cls]["precision"]
        r = report[cls]["recall"]
        f = report[cls]["f1-score"]
        print(f"{cls:<20}{p:<15.4f}{r:<15.4f}{f:<15.4f}")

    # Weighted averages
    print("\n" + "-" * 65)
    print(f"{'Weighted Average':<20}", end="")
    print(f"{report['weighted avg']['precision']:<15.4f}", end="")
    print(f"{report['weighted avg']['recall']:<15.4f}", end="")
    print(f"{report['weighted avg']['f1-score']:<15.4f}")
    print(f"{'Accuracy':<20}{report['accuracy']:<15.4f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)


def main():
    """Main training pipeline."""
    # Load and preprocess data
    print("Loading dataset...")
    df_raw = load_dataset("sleep_quality.csv")
    df_processed = preprocess_data_basic(df_raw)

    # Save processed data
    df_processed.to_csv("sleep_processed_bayes.csv", index=False)
    print(f"Processed dataset saved: {len(df_processed)} samples")

    # Exploratory visualizations
    plot_class_distribution(df_processed)
    plot_correlation_matrix(df_processed, exclude_cols=["Person ID", "Sleep Disorder"])

    # KDE-based Naive Bayes - Multiclass
    print("\n" + "=" * 60)
    print("KDE-BASED NAIVE BAYES CLASSIFIER")
    print("=" * 60)

    bayes_kde = BayesClassifier()
    evaluate_classifier(
        df_processed,
        "Sleep Disorder",
        bayes_kde,
        title="KDE Bayes - Multiclass Classification",
    )

    # Binary classification
    df_binary = df_processed.copy()
    df_binary.loc[
        (df_processed["Sleep Disorder"] == "Insomnia")
        | (df_processed["Sleep Disorder"] == "Sleep Apnea"),
        "Sleep Disorder",
    ] = "Disorder"

    evaluate_classifier(
        df_binary,
        "Sleep Disorder",
        BayesClassifier(),
        title="KDE Bayes - Binary Classification (Disorder vs No Disorder)",
    )

    # Gaussian-based Naive Bayes
    print("\n" + "=" * 60)
    print("GAUSSIAN-BASED NAIVE BAYES CLASSIFIER")
    print("=" * 60)

    bayes_gaussian = BayesGaussianClassifier()
    evaluate_classifier(
        df_processed,
        "Sleep Disorder",
        bayes_gaussian,
        title="Gaussian Bayes - Multiclass Classification",
    )

    # Gaussian Bayes - Binary classification
    evaluate_classifier(
        df_binary,
        "Sleep Disorder",
        BayesGaussianClassifier(),
        title="Gaussian Bayes - Binary Classification (Disorder vs No Disorder)",
    )

    plt.show()


if __name__ == "__main__":
    main()
