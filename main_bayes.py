import pandas as pd
import numpy as np
import KunstlicheIntel as ki
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, ShuffleSplit

sleep = pd.read_csv("sleep_quality.csv")
classes = "Sleep Disorder"

# ==== Data preprocessing ====
sleep[["Systolic Pressure", "Diastolic Pressure"]] = sleep["Blood Pressure"].str.split("/", expand=True).astype(int)
sleep["Sleep Disorder"] = sleep["Sleep Disorder"].fillna("No Disorder")
sleep = sleep.drop(columns=["Blood Pressure"])
sleep.loc[sleep["BMI Category"] == "Normal Weight", "BMI Category"] = "Normal" # Author of dataset left different labels, which can be substituted by one
if len(sleep["Person ID"].unique()) == len(sleep):
    print("Values in column 'Person ID' are unique and simultaneously not necessary in machine learning")
    sleep = sleep.drop(columns=["Person ID"])

# sleep.to_csv("sleepProcessed.csv",index=False)

drop_columns = pd.read_csv('columns_to_drop.csv')['columns_to_drop'].tolist()
sleep = sleep.drop(columns = drop_columns)


# ==== Splitting data for machine learning ====
sleep = ki.DataProcessing.shuffle(sleep)
sleep_train, sleep_test = ki.DataProcessing.split_train_test(sleep, 0.7, classes)

# ==== Machine learning ==== 
print()
print("Machine learning standard")
clf = ki.BayesClassificator()
clf.fit(sleep, classes)

sleep_pred = clf.predict(sleep_test)

print(classification_report(sleep_test[classes], sleep_pred[classes]))

abstracts = sorted(sleep_test[classes].unique())
cm = confusion_matrix(sleep_test[classes], sleep_pred[classes], labels=abstracts)
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
print(classification_report(sleep_bin_test[classes], sleep_bin_pred[classes]))

abstracts = sorted(sleep_bin_test[classes].unique())
cm = confusion_matrix(sleep_bin_test[classes], sleep_bin_pred[classes], labels=abstracts)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=abstracts, yticklabels=abstracts)
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
print(classification_report(sleep_gauss_test[classes], sleep_gauss_pred[classes]))

abstracts = sorted(sleep_gauss_test[classes].unique())
cm = confusion_matrix(sleep_gauss_test[classes], sleep_gauss_pred[classes], labels=abstracts)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=abstracts, yticklabels=abstracts)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('Confusion Matrix')
plt.show()