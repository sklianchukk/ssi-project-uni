import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, ShuffleSplit
from Bayes_sklearn import BayesSklearnWrapper

sleep = pd.read_csv("sleepProcessed.csv")
classes = "Sleep Disorder"
abstracts = sleep["Sleep Disorder"].unique()

bmi_mapping = {
    "Normal": 0,
    "Overweight": 1,
    "Obese": 2
}

sleep_numerical = sleep.select_dtypes(include=[np.number]).copy()
sleep_numerical["BMI Category"] = sleep["BMI Category"].map(bmi_mapping)
sleep_numerical["Gender"] = sleep["Gender"].map({"Male":1, "Female":0})
corr_matrix = sleep_numerical.corr(method='spearman')
corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
thresholds = corr_matrix.where(corr_matrix.abs() > 0.75).stack().dropna().abs().tolist()
thresholds.append(0.75)
thresholds.append(1.0)
print(thresholds)


# --- PĘTLA WALIDACJI KRZYŻOWEJ ---
results_list = []
cv_strategy = ShuffleSplit(n_splits=25, test_size=0.3, random_state=42)

for th in thresholds:
    # Wyznaczanie kolumn do usunięcia
    mask = (((corr_matrix.abs() > th).any(axis=1)) & (~corr_matrix.isna().all(axis=1)))
    dropColumn = corr_matrix[mask].index.tolist()
    
    # Przygotowanie X i y
    sleep_temp = sleep.drop(columns=dropColumn)
    X = sleep_temp.drop(columns=[classes])
    y = sleep_temp[classes]
    
    # Uruchomienie walidacji
    wrapper = BayesSklearnWrapper(target_column=classes)
    cv_results = cross_validate(
        wrapper, X, y, 
        cv=cv_strategy, 
        scoring='f1_weighted', 
        n_jobs=-1
    )
    
    # Zbieranie wyników do listy (bezpieczniej niż słownik z kluczem F1)
    mean_f1 = cv_results['test_score'].mean()
    results_list.append({
        'f1': mean_f1,
        'dropped': dropColumn,
        'threshold': th
    })

# Sortowanie i wybór najlepszego zestawu
results_list.sort(key=lambda x: x['f1'], reverse=True)
best_result = results_list[0]

print(f"Najlepszy F1-score: {best_result['f1']:.4f} dla progu {best_result['threshold']}")

# Zapis kolumn do usunięcia
pd.DataFrame({'columns_to_drop': best_result['dropped']}).to_csv('columns_to_drop.csv', index=False)