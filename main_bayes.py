import pandas as pd

sleep = pd.read_csv("sleep_quality.csv")
classes = "Sleep Disorder"

# ==== Data preprocessing ====
sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")
sleep = sleep.drop(columns=["Blood Pressure"])
map_bmi_cat = {
    "Normal Weight": "Normal",
    "Obese": "Overweight"
}
sleep["BMI Category"].map(map_bmi_cat) # Author of dataset left different labels, which can be substituted by one
if len(sleep["Person ID"].unique()) == len(sleep):
    print("Values in column 'Person ID' are unique and simultaneously not necessary in machine learning")
    sleep = sleep.drop(columns=["Person ID"])

sleep.to_csv("sleepProcessed.csv",index=False)