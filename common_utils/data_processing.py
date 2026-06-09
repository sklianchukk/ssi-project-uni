import pandas as pd
import numpy as np


def load_dataset(file_name: str) -> pd.DataFrame:
    """Load dataset from CSV file with error handling."""
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"File '{file_name}' not found. Check filename."
        ) from e
    except Exception as e:
        raise Exception(f"Error reading file '{file_name}': {e}") from e
    return df


def preprocess_data_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: split blood pressure, fill missing values, standardize BMI categories."""
    df_processed = df.copy()

    # Split blood pressure into Systolic and Diastolic
    df_processed[["Systolic Pressure", "Diastolic Pressure"]] = (
        df_processed["Blood Pressure"].str.split("/", expand=True).astype(int)
    )
    df_processed = df_processed.drop(columns=["Blood Pressure"])

    # Fill missing Sleep Disorder values
    df_processed["Sleep Disorder"] = df_processed["Sleep Disorder"].fillna("No Disorder")

    # Standardize BMI categories
    bmi_mapping = {"Normal Weight": "Normal", "Obese": "Overweight"}
    df_processed["BMI Category"] = df_processed["BMI Category"].map(
        lambda x: bmi_mapping.get(x, x)
    )

    # Remove Person ID if all unique (no predictive value)
    if len(df_processed["Person ID"].unique()) == len(df_processed):
        df_processed = df_processed.drop(columns=["Person ID"])

    return df_processed


def preprocess_data_for_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing for Random Forest: rename BMI, map gender, extract diastolic pressure."""
    df_processed = df.copy()

    # Rename and standardize BMI Category
    df_processed = df_processed.rename(columns={"BMI Category": "BMI"})
    df_processed["BMI"] = df_processed["BMI"].replace(
        {"Normal Weight": "Normal", "Obese": "Overweight"}
    )

    # Map gender to binary (1=Male, 0=Female)
    df_processed["Gender"] = df_processed["Gender"].map({"Male": 1, "Female": 0})

    # Standardize Sleep Disorder values
    df_processed["Sleep Disorder"] = (
        df_processed["Sleep Disorder"].fillna("No Disorder").replace({"None": "No Disorder"})
    )

    # Extract diastolic pressure only
    df_processed["Diastolic Pressure"] = pd.to_numeric(
        df_processed["Blood Pressure"].str.split("/", expand=True)[1]
    )
    df_processed = df_processed.drop("Blood Pressure", axis=1)

    return df_processed
