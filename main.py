from data_processing import load_dataset, prepare_data
from evaluation import (
    analyze_correlations,
    train_and_evaluate,
    plot_class_distribution,
)

if __name__ == "__main__":
    df_raw = load_dataset("sleep_quality.csv")
    df_prepared = prepare_data(df_raw)

    plot_class_distribution(df_prepared)
    analyze_correlations(df_prepared)

    print("MULTICLASS CLASSIFICATION")
    train_and_evaluate(df_prepared)

    print("\nBINARY CLASSIFICATION")
    train_and_evaluate(df_prepared, binary=True)
