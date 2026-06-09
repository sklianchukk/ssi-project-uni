import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def ensure_output_folder(folder_path: str = "Evaluation_images") -> str:
    """Create output folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def plot_class_distribution(df: pd.DataFrame, target_column: str = "Sleep Disorder", folder: str = None) -> None:
    """Plot and save class distribution chart."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_column, palette="Set2")
    plt.title("Class Distribution of Sleep Disorders")
    plt.xlabel("Sleep Disorder Category")
    plt.ylabel("Count")
    plt.tight_layout()

    if folder:
        filepath = os.path.join(folder, "class_distribution.png")
        plt.savefig(filepath, dpi=300)

    plt.show(block=False)
    plt.pause(1)


def plot_correlation_matrix(df: pd.DataFrame, exclude_cols: list = None, folder: str = None) -> None:
    """Plot correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if exclude_cols:
        numeric_df = numeric_df.drop(exclude_cols, axis=1, errors="ignore")

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()

    if folder:
        filepath = os.path.join(folder, "correlation_matrix.png")
        plt.savefig(filepath, dpi=300)

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
    folder: str = None,
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

    if folder:
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath, dpi=300)
    else:
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
    folder: str = None,
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

    if folder:
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath, dpi=300)
    else:
        plt.savefig(filename, dpi=300)

    plt.show(block=False)
    plt.pause(1)


def print_gridsearch_results(grid_search_obj, top_n: int = 5) -> None:
    """Print GridSearchCV results with best parameters and top configurations."""
    print("\n" + "=" * 70)
    print("GRIDSEARCH OPTIMIZATION RESULTS")
    print("=" * 70)

    # Best parameters and score
    print(f"\nBest F1-Score (weighted): {grid_search_obj.best_score_:.4f}")
    print("\nBest Parameter Combination:")
    print("-" * 70)
    for param_name, param_value in grid_search_obj.best_params_.items():
        clean_name = param_name.replace("classifier__", "")
        print(f"  {clean_name:<30} {param_value}")

    # Top configurations
    results_df = pd.DataFrame(grid_search_obj.cv_results_)
    results_df["rank"] = results_df["rank_test_score"]

    print(f"\nTop {min(top_n, len(results_df))} Parameter Combinations:")
    print("-" * 70)

    top_results = results_df.nsmallest(top_n, "rank_test_score")

    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"\n#{idx} - F1-Score: {row['mean_test_score']:.4f} " f"(±{row['std_test_score']:.4f})")

        # Extract and print parameters
        param_keys = [k for k in results_df.columns if k.startswith("param_")]
        for param_key in sorted(param_keys):
            if pd.notna(row[param_key]):
                clean_name = param_key.replace("param_classifier__", "")
                print(f"     {clean_name:<28} {row[param_key]}")

    print("\n" + "=" * 70)
