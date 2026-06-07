import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, ShuffleSplit
from Bayes_sklearn import BayesSklearnWrapper

# Load preprocessed sleep disorder dataset
sleep = pd.read_csv("sleepProcessed.csv")
classes = "Sleep Disorder"
abstracts = sleep["Sleep Disorder"].unique()

# Create numerical encoding for categorical features
bmi_mapping = {
    "Normal": 0,
    "Overweight": 1,
    "Obese": 2
}

# Convert categorical columns to numerical for correlation analysis
sleep_numerical = sleep.select_dtypes(include=[np.number]).copy()
sleep_numerical["BMI Category"] = sleep["BMI Category"].map(bmi_mapping)
sleep_numerical["Gender"] = sleep["Gender"].map({"Male":1, "Female":0})

# Calculate Spearman correlation matrix and extract upper triangle (no duplicates)
corr_matrix = sleep_numerical.corr(method='spearman')
corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
print(corr_matrix)

# Extract all correlation thresholds above 0.75 for feature selection testing
thresholds = corr_matrix.where(corr_matrix.abs() > 0.75).stack().dropna().abs().tolist()
thresholds.append(0.75)
print(thresholds)

# Shuffle-split cross-validation: dataset randomly split 70/30 for 25 iterations
results_list = []
cv_strategy = ShuffleSplit(n_splits=25, test_size=0.3, random_state=42)

# Test each correlation threshold to find optimal feature set
for th in thresholds:
    # Identify highly correlated features to drop (correlation > threshold)
    mask = (((corr_matrix.abs() > th).any(axis=1)) & (~corr_matrix.isna().all(axis=1)))
    dropColumn = corr_matrix[mask].index.tolist()

    # Prepare features (X) and target (y) after dropping correlated features
    sleep_temp = sleep.drop(columns=dropColumn)
    X = sleep_temp.drop(columns=[classes])
    y = sleep_temp[classes]

    # Train and evaluate classifier using cross-validation
    wrapper = BayesSklearnWrapper(target_column=classes)
    cv_results = cross_validate(
        wrapper, X, y,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=-1
    )

    # Store F1-score and configuration for comparison
    mean_f1 = cv_results['test_score'].mean()
    results_list.append({
        'f1': mean_f1,
        'dropped': dropColumn,
        'threshold': th
    })

# Find configuration with best F1-score
results_list.sort(key=lambda x: x['f1'], reverse=True)
best_result = results_list[0]

print(f"Best F1-score: {best_result['f1']:.4f} for threshold {best_result['threshold']}")

# Save optimal feature set to file for later use
pd.DataFrame({'columns_to_drop': best_result['dropped']}).to_csv('columns_to_drop.csv', index=False)