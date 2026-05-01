import pandas as pd
import numpy as np
import KunstlicheIntel as ki
import seaborn as sns
import matplotlib.pyplot as plt

sleep = pd.read_csv("sleep_quality.csv")
print(sleep.describe())


sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")
sleep = sleep.drop(columns=["Blood Pressure"])
print(sleep.isna().sum())

print(sleep.head(10))

# sns.pairplot(sleep[["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate","Daily Steps", "Systolic Pressure", "Diastolic Pressure","Sleep Disorder"]],hue="Sleep Disorder")
# plt.show()
