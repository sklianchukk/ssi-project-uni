import pandas as pd

# Load raw sleep quality dataset
sleep = pd.read_csv("sleep_quality.csv")
classes = "Sleep Disorder"

# Split combined blood pressure column into separate systolic and diastolic measurements
sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)

# Fill missing sleep disorder values with "No Disorder" label
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")

# Drop original blood pressure column after splitting
sleep = sleep.drop(columns=["Blood Pressure"])

# Map inconsistent BMI category labels to standardized values
map_bmi_cat = {
    "Normal Weight": "Normal",
    "Obese": "Overweight"
}
sleep["BMI Category"].map(map_bmi_cat)

# Remove Person ID if it contains only unique values (no predictive value for classification)
if len(sleep["Person ID"].unique()) == len(sleep):
    print("Values in column 'Person ID' are unique and not necessary for machine learning")
    sleep = sleep.drop(columns=["Person ID"])

# Save processed dataset for model training
sleep.to_csv("sleepProcessed.csv", index=False)
