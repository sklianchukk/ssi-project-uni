import pandas as pd


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

    # rename bmi category column to bmi
    df_data = df_data.rename(columns={"BMI Category": "BMI"})

    # replace specific bmi values
    df_data["BMI"] = df_data["BMI"].replace(
        {"Normal Weight": "Normal", "Obese": "Overweight"}
    )

    # map gender text values to numeric binary representations
    df_data["Gender"] = df_data["Gender"].map({"Male": 1, "Female": 0})

    # fill missing values and unify text representations
    df_data["Sleep Disorder"] = (
        df_data["Sleep Disorder"].fillna("No Disorder").replace({"None": "No Disorder"})
    )

    # extract only diastolic pressure to avoid multicollinearity with systolic pressure
    df_data["Diastolic Pressure"] = pd.to_numeric(
        df_data["Blood Pressure"].str.split("/", expand=True)[1]
    )

    # remove the obsolete source column
    df_data = df_data.drop("Blood Pressure", axis=1)

    return df_data

