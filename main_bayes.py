import pandas as pd
import numpy as np
import KunstlicheIntel as ki
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

sleep = pd.read_csv("sleep_quality.csv")
classes = "Sleep Disorder"

# ==== Data preprocessing ====
sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")
sleep = sleep.drop(columns=["Blood Pressure"])
sleep.loc[sleep["BMI Category"] == "Normal Weight", "BMI Category"] = "Normal" # Author of dataset left different labels, which can be substituted by one
print(sleep["BMI Category"].unique())
if len(sleep["Person ID"].unique()) == len(sleep):
    print("Values in column 'Person ID' are unique and simultaneously not necessary in machine learning")
    sleep = sleep.drop(columns=["Person ID"])

# sns.pairplot(sleep[["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate","Daily Steps", "Systolic Pressure", "Diastolic Pressure","Sleep Disorder"]],hue="Sleep Disorder")
# plt.show()

threshold = 0.8
sleep_numerical = sleep.select_dtypes(include=[np.number])
corr_matrix = sleep_numerical.corr()
corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
dropColumns = [col for col in corr_matrix.columns.tolist()[::-1] if corr_matrix[col].abs().max() > threshold and not corr_matrix[col].isna().all()]
print(corr_matrix)
sleep = sleep.drop(columns = dropColumns)


# ==== Splitting data for machine learning ====
sleep = ki.DataProcessing.shuffle(sleep)
sleep_train, sleep_test = ki.DataProcessing.split_train_test(sleep, 0.7, classes)

# ==== Machine learning ==== 
print()
print("Machine learning standard")
clf = ki.BayesClassificator()
clf.fit(sleep, classes)

sleep_pred = clf.predict(sleep_test)
abstracts = sleep[classes].unique()

print(classification_report(sleep_test[classes], sleep_pred[classes], target_names=abstracts))

cm = confusion_matrix(sleep_test[classes], sleep_pred[classes])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=abstracts, yticklabels=abstracts)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')



# ==== Binary Machine Learning(Insomnia and Apnea combined as Disorder) ====

print()
print("Binary Machine Learning(Insomnia and Apnea combined as Disorder)")
sleep_bin = sleep.copy()

sleep_bin.loc[(sleep["Sleep Disorder"] == "Insomnia") | (sleep["Sleep Disorder"] == "Sleep Apnea"), "Sleep Disorder"] = "Disorder"
sleep_bin = ki.DataProcessing.shuffle(sleep_bin)
sleep_bin_train, sleep_bin_test = ki.DataProcessing.split_train_test(sleep_bin, 0.7, classes)

clf_bin = ki.BayesClassificator()
clf_bin.fit(sleep_bin_train, classes)
sleep_bin_pred = clf_bin.predict(sleep_bin_test)
abstracts_bin = sleep_bin["Sleep Disorder"].unique()
print(classification_report(sleep_bin_test[classes], sleep_bin_pred[classes], target_names=abstracts_bin))

cm = confusion_matrix(sleep_bin_test[classes], sleep_bin_pred[classes])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=abstracts_bin, yticklabels=abstracts_bin)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')



# ==== Machine learning with super Naive Bayes ====

print()
print("Machine learning with super Naive Bayes")
sleep_gauss_train, sleep_gauss_test = ki.DataProcessing.split_train_test(sleep, 0.7, classes)
clf_gauss = ki.BayesGuassianClassificator()

clf_gauss.fit(sleep_gauss_train, classes)
sleep_gauss_pred = clf_gauss.predict(sleep_gauss_test)
print(classification_report(sleep_gauss_test[classes], sleep_gauss_pred[classes], target_names=abstracts))

cm = confusion_matrix(sleep_gauss_test[classes], sleep_gauss_pred[classes])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=abstracts, yticklabels=abstracts)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')
plt.show()