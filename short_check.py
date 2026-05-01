import pandas as pd

sleep = pd.read_csv("sleep_quality.csv")
sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")
sleep = sleep.drop(columns=["Blood Pressure"])
sleep.loc[sleep["BMI Category"] == "Normal Weight", "BMI Category"] = "Normal"

for col in sleep.select_dtypes(exclude=['number']).columns:
    print(sleep[col].unique())