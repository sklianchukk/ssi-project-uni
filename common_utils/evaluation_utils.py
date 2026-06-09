import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def plot_class_distribution(df: pd.DataFrame, target_column: str = "Sleep Disorder") -> None:
    """Plot and save class distribution chart."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_column, palette="Set2")
    plt.title("Class Distribution of Sleep Disorders")
    plt.xlabel("Sleep Disorder Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300)
    plt.show(block=False)
    plt.pause(1)


def plot_correlation_matrix(df: pd.DataFrame, exclude_cols: list = None) -> None:
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if exclude_cols:
        numeric_df = numeric_df.drop(exclude_cols, axis=1, errors="ignore")

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)


def print_metrics(
    y_true,
    y_pred,
    metric_names: list = None,
    cv_results: dict = None,
    label: str = "",
) -> None:
    """Print classification metrics and CV results."""
    if label:
        print(f"\n{label}")
        print("=" * 50)

    if cv_results:
        print("\nAverage Scores from Cross-Validation:")
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                metric_name = metric.replace("test_", "").replace("_weighted", "").title()
                print(f"{metric_name}: {np.mean(scores):.4f}")

    if y_true is not None and y_pred is not None:
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))


def plot_confusion_matrix(
    y_true,
    y_pred,
    filename: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
    display_labels: list = None,
    rotation: int = 45,
) -> None:
    """Plot and save confusion matrix."""
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        display_labels=display_labels,
        xticks_rotation=rotation,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show(block=False)
    plt.pause(1)


def print_feature_importances(
    feature_names: list,
    importances: np.ndarray,
    indices: np.ndarray,
    top_n: int = 8,
    title: str = "Top Feature Importances",
) -> None:
    """Print top feature importances."""
    print(f"\n{title}:")
    print("-" * 40)

    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")


def plot_feature_importances(
    feature_names: list,
    importances: np.ndarray,
    indices: np.ndarray,
    filename: str = "feature_importance.png",
    title: str = "Top Feature Importances",
    top_n: int = 8,
) -> None:
    """Plot and save feature importances."""
    plt.figure(figsize=(10, 6))
    top_n = min(top_n, len(feature_names))
    features = [feature_names[i] for i in indices[:top_n]]
    scores = importances[indices[:top_n]]

    sns.barplot(x=scores, y=features, palette="viridis")
    plt.title(title)
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show(block=False)
    plt.pause(1)
