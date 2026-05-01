import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# definiuje podział cech na kategorie dla transformatora
CATEGORICAL_FEATURES = ["Gender", "Occupation", "BMI"]
NUMERIC_FEATURES = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Systolic Pressure",
    "Diastolic Pressure",
    "Heart Rate",
    "Daily Steps",
]


def load_dataset(file_name: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file with error handling."""
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"File '{file_name}' does not exist! Check the file's name whether it's correct."
        ) from e
    except Exception as e:
        raise Exception(f"Error reading file '{file_name}': {e}") from e

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the initial dataset."""
    df_data = df.copy()

    # zmienia nazwę kolumny BMI Category na BMI
    df_data = df_data.rename(columns={"BMI Category": "BMI"})
    # zastępuje wartości BMI
    df_data["BMI"] = df_data["BMI"].replace(
        {"Normal Weight": "Normal", "Obese": "Overweight"}
    )

    # mapuje wartości tekstowe Gender na liczby
    df_data["Gender"] = df_data["Gender"].map({"Male": 1, "Female": 0})

    # uzupełnia i ujednolica wartości zaburzeń snu
    df_data["Sleep Disorder"] = (
        df_data["Sleep Disorder"].fillna("No Disorder").replace({"None": "No Disorder"})
    )

    # rozdziela tekstową kolumnę z ciśnieniem na dwie osobne wartości numeryczne
    df_data[["Systolic Pressure", "Diastolic Pressure"]] = df_data[
        "Blood Pressure"
    ].str.split("/", expand=True)

    # konwertuje kolumny ciśnienia na liczby
    df_data["Systolic Pressure"] = pd.to_numeric(df_data["Systolic Pressure"])
    df_data["Diastolic Pressure"] = pd.to_numeric(df_data["Diastolic Pressure"])

    # usuwa oryginalną kolumnę Blood Pressure
    df_data = df_data.drop("Blood Pressure", axis=1)

    return df_data


def analyze_correlations(df: pd.DataFrame) -> None:
    """Plots a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def create_preprocessor() -> ColumnTransformer:
    """Creates a column transformer for categorical and numeric features."""
    # definiuje podział cech na kategorie dla transformatora
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        # jeśli trafiamy na kolumnę z liczbami, to zostawiamy bez zmian
        remainder="passthrough",
    )


def create_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Creates a pipeline with preprocessor and random forest classifier."""
    # potoki są używane, żeby zapobiec wycieku danych
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                ),
            ),
        ]
    )


def get_feature_importances(pipeline: Pipeline) -> tuple:
    """Extracts feature names and importance scores from trained pipeline."""
    # docieramy do narzędzia przetwarzającego
    rf_model = pipeline.named_steps["classifier"]
    # preprocessor utworzył nowe kolumny jak np. Occupation_Nurse, a więc pobieramy je
    cat_feature_names = (
        pipeline.named_steps["preprocessor"]
        .transformers_[0][1]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_feature_names = list(cat_feature_names) + NUMERIC_FEATURES
    importances = rf_model.feature_importances_
    # sortujemy indeksy cech na podstawie ich istotności dla modelu w porządku malejącym
    indices = np.argsort(importances)[::-1]
    
    return all_feature_names, importances, indices


def print_feature_importances(
    all_feature_names: list, importances: np.ndarray, indices: np.ndarray, title: str = "Top 5 Feature Importances"
) -> None:
    """Prints top 5 most important features."""
    print(f"\n{title}:")
    # sortujemy i pokazujemy 5 najważniejszych cech, które dały nam najlepszy podział
    for f in range(min(5, len(all_feature_names))):
        print(f"{all_feature_names[indices[f]]}: {importances[indices[f]]:.4f}")


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Confusion Matrix",
    display_labels: list = None,
) -> None:
    """Evaluates model and displays confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    if display_labels:
        print(
            f"\nClassification Report:\n",
            classification_report(y_test, y_pred, target_names=display_labels),
        )
    else:
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # generowanie macierzy pomyłek w formie graficznej
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        display_labels=display_labels,
        xticks_rotation=45 if not display_labels else 0,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return y_pred


def train_and_evaluate(
    df: pd.DataFrame,
    binary: bool = False
) -> None:
    """Trains and evaluates model (multiclass or binary classification)."""
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    
    if binary:
        y = df["Sleep Disorder"].apply(lambda x: 0 if x == "No Disorder" else 1)
        display_labels = ["No Disorder", "Disorder"]
        title = "Binary Confusion Matrix"
        feature_title = "Top 5 Feature Importances (Binary Classification)"
    else:
        y = df["Sleep Disorder"]
        display_labels = None
        title = "Confusion Matrix"
        feature_title = "Top 5 Feature Importances"

    # zachowuje identyczne proporcje klas w obu zbiorach dzięki parametrowi stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    preprocessor = create_preprocessor()
    pipeline = create_pipeline(preprocessor)
    
    pipeline.fit(X_train, y_train)

    # wywołuje sam model
    evaluate_model(pipeline, X_test, y_test, title=title, display_labels=display_labels)

    # pokazuje najważniejsze parametry
    all_feature_names, importances, indices = get_feature_importances(pipeline)
    print_feature_importances(all_feature_names, importances, indices, title=feature_title)


if __name__ == "__main__":
    df_raw = load_dataset("sleep_quality.csv")
    df_prepared = prepare_data(df_raw)

    analyze_correlations(df_prepared)
    
    print("MULTICLASS CLASSIFICATION")
    train_and_evaluate(df_prepared)
    
    print("\nBINARY CLASSIFICATION")
    train_and_evaluate(df_prepared, binary=True)
