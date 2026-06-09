from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from Bayes_sklearn import BayesSklearnWrapper
import numpy as np
import KunstlicheIntel as ki

# Load preprocessed sleep disorder dataset
sleep = pd.read_csv('sleepProcessed.csv')
classes = "Sleep Disorder"
drop_columns = pd.read_csv('columns_to_drop.csv')['columns_to_drop'].tolist()
sleep = sleep.drop(columns = drop_columns)

# Stratified k-fold cross-validation: ensures each fold has same class proportions
partition_cv = StratifiedKFold(n_splits=5, shuffle=True)


def ML_stats_mean(dataset, classes, clf):
    # Wrap custom classifier to comply with scikit-learn API
    wrappedClf = BayesSklearnWrapper(target_column=classes)
    wrappedClf.model = clf
    y_true = dataset[classes]

    # Perform cross-validation: train on 4 folds, predict on held-out fold
    y_pred_cv = cross_val_predict(
        wrappedClf,
        dataset.drop(columns=classes),
        y_true,
        cv=partition_cv,
        n_jobs=-1
    )

    # Calculate evaluation metrics
    cm = confusion_matrix(y_true, y_pred_cv)

    report = classification_report(y_true, y_pred_cv, output_dict=True)
    abstracts = sorted(y_true.unique().tolist())

    # Extract metrics per class
    f1_w = report['weighted avg']['f1-score']
    f1 = [report[ab]['f1-score'] for ab in abstracts]
    precision_w = report['weighted avg']['precision']
    precision = [report[ab]['precision'] for ab in abstracts]
    recall_w = report['weighted avg']['recall']
    recall = [report[ab]['recall'] for ab in abstracts]
    accuracy = report['accuracy']
    columns = ['precision', 'recall', 'f1-score']

    # Print per-class metrics
    print(f"{(' '):<12}{(columns[0]):<12}{(columns[1]):<12}{(columns[2]):<12}")
    for i, p, r, f in zip(abstracts, precision, recall, f1):
        print(f"{(i):<12}{np.round(p, 2):<12}{np.round(r, 2):<12}{np.round(f, 2):<12}")

    # Print weighted averages
    print(f"\nAvarages:")
    print(f"F1-score (weighted): {f1_w:.4f}")
    print(f"Precision (weighted): {precision_w:.4f}")
    print(f"Recall (weighted): {recall_w:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=abstracts)

# Test KDE-based Naive Bayes on multi-class sleep disorders
disp = ML_stats_mean(sleep, classes, ki.BayesClassificator())
disp.plot(cmap=plt.cm.Blues)


# Test on binary classification: combine Insomnia and Sleep Apnea as single "Disorder" class
print()
print("Binary classification (Insomnia and Sleep Apnea combined as Disorder)")

sleep_bin = sleep.copy()
sleep_bin.loc[(sleep["Sleep Disorder"] == "Insomnia") | (sleep["Sleep Disorder"] == "Sleep Apnea"), "Sleep Disorder"] = "Disorder"

disp = ML_stats_mean(sleep_bin, classes, ki.BayesClassificator())
disp.plot(cmap=plt.cm.Blues)

# Test Gaussian-based Naive Bayes on multi-class problem
print()
print("Gaussian-based Naive Bayes classifier")
disp = ML_stats_mean(sleep, classes, ki.BayesGuassianClassificator())
disp.plot(cmap=plt.cm.Blues)
plt.show()