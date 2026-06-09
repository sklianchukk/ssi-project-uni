# Quick Start Guide

## Project Overview

Your project has been reorganized into a clean, modular structure separating concerns between two ML approaches:
- **Naive Bayes models** (KDE and Gaussian variants)
- **Random Forest classifier**

## 📁 New Directory Structure

```
project/
├── common_utils/              # Shared code
│   ├── config.py             # Feature definitions
│   ├── data_processing.py    # Data loading & preprocessing
│   ├── evaluation_utils.py   # Visualization & metrics
│   └── __init__.py
├── bayes_model/              # Naive Bayes implementation
│   ├── classifier.py         # KDE and Gaussian classifiers
│   ├── sklearn_wrapper.py    # Sklearn API wrapper
│   ├── feature_selection.py  # Correlation-based feature selection
│   └── __init__.py
├── random_forest_model/      # Random Forest implementation
│   ├── pipeline.py           # Pipeline & feature importance
│   └── __init__.py
├── main_bayes.py            # Run this for Naive Bayes
├── main_random_forest.py    # Run this for Random Forest
└── Documentation files
```

## ⚡ Quick Start

### 1. Run Naive Bayes Classification
```bash
python main_bayes.py
```

Trains and evaluates:
- **KDE-based Naive Bayes** (multiclass + binary)
- **Gaussian Naive Bayes** (multiclass)

Output: Confusion matrices, classification reports, `sleep_processed_bayes.csv`

### 2. Run Random Forest Classification
```bash
python main_random_forest.py
```

Trains and evaluates:
- **Random Forest with GridSearchCV** (multiclass + binary)

Output: Confusion matrices, feature importance plots, `sleep_processed_rf.csv`

## 📊 What You Get

Both scripts produce:
- Class distribution visualization
- Correlation matrix heatmap
- Confusion matrices (multiclass & binary)
- Classification reports (precision, recall, F1-score)
- Feature importance analysis
- Cross-validation metrics

## 🔑 Key Features

| Aspect | Bayes | Random Forest |
|--------|-------|---------------|
| **Preprocessing** | Splits blood pressure | Gender to binary, extracts diastolic |
| **Validation** | 5-fold stratified K-fold | 5-fold + GridSearchCV |
| **Feature Selection** | Correlation thresholds (0.75) | Built-in feature importance |
| **Output** | Per-class metrics | Feature importance ranking |

## 🚀 Advanced Usage

### Custom Bayes Classification
```python
from bayes_model.classifier import BayesClassifier
from common_utils.data_processing import load_dataset, preprocess_data_basic

df = load_dataset("sleep_quality.csv")
df = preprocess_data_basic(df)

clf = BayesClassifier()
clf.fit(df, "Sleep Disorder")
predictions = clf.predict(df.drop("Sleep Disorder", axis=1))
```

### Custom Random Forest with Cross-Validation
```python
from sklearn.model_selection import cross_validate
from random_forest_model.pipeline import create_preprocessor, create_pipeline
from common_utils.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

pipeline = create_pipeline(create_preprocessor())

X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df["Sleep Disorder"]

cv_results = cross_validate(pipeline, X, y, cv=5, scoring="f1_weighted")
```

## 📋 Old Files to Delete (After Testing)

These files have been merged into the new structure:
- `KunstlicheIntel.py`
- `Bayes_sklearn.py`
- `data_processing.py` (old version)
- `data_processing_random_forest.py`
- `detectThreshold.py`
- `evaluation.py`
- `train_predict.py`
- `main.py`
- `model.py`
- `config.py` (old version)

## 📖 Documentation

- **PROJECT_STRUCTURE.md** - Detailed file descriptions and API docs
- **MIGRATION_GUIDE.md** - Old → new file mapping and import changes

## ✅ What's Preserved

✓ KDE-based Naive Bayes functionality
✓ Gaussian Naive Bayes functionality
✓ Random Forest with GridSearchCV
✓ All evaluation metrics and visualizations
✓ Feature selection and importance
✓ Binary and multiclass support

## 💡 Next Steps

1. **Test both models**:
   ```bash
   python main_bayes.py
   python main_random_forest.py
   ```

2. **Verify outputs** - Check generated CSV and PNG files

3. **Delete old files** - Remove files listed above

4. **Commit to git** - New organized structure is ready!

---

**Questions?** Check PROJECT_STRUCTURE.md for detailed API documentation.
