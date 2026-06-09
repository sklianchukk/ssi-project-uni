# Sleep Quality Classification Project

Refactored machine learning project for sleep disorder classification using two different approaches:
- **Naive Bayes**: KDE-based and Gaussian-based classifiers
- **Random Forest**: Scikit-learn RandomForestClassifier with GridSearchCV

## Project Structure

```
.
├── common_utils/                  # Shared utilities and configuration
│   ├── config.py                 # Feature definitions and constants
│   ├── data_processing.py        # Data loading and preprocessing
│   ├── evaluation_utils.py       # Visualization and metrics functions
│   └── __init__.py
│
├── bayes_model/                   # Naive Bayes implementation
│   ├── classifier.py             # BayesClassifier (KDE) and BayesGaussianClassifier
│   ├── sklearn_wrapper.py        # Sklearn API wrapper for Bayes models
│   ├── feature_selection.py      # Optimal feature finding with correlation thresholds
│   └── __init__.py
│
├── random_forest_model/           # Random Forest implementation
│   ├── pipeline.py               # Pipeline creation and feature importance extraction
│   └── __init__.py
│
├── main_bayes.py                 # Main training script for Naive Bayes
├── main_random_forest.py         # Main training script for Random Forest
├── requirements.txt              # Python dependencies
└── sleep_quality.csv             # Raw dataset
```

## Data Preprocessing

The project supports two different preprocessing pipelines:

### Basic Preprocessing (Naive Bayes)
Located in: `common_utils/data_processing.py` → `preprocess_data_basic()`

- Splits blood pressure into Systolic and Diastolic
- Fills missing Sleep Disorder values with "No Disorder"
- Standardizes BMI categories
- Removes Person ID if all unique values

**Output**: `sleep_processed_bayes.csv`

### Random Forest Preprocessing
Located in: `common_utils/data_processing.py` → `preprocess_data_for_random_forest()`

- Renames "BMI Category" to "BMI"
- Maps gender to binary (1=Male, 0=Female)
- Extracts only diastolic pressure
- Standardizes Sleep Disorder values

**Output**: `sleep_processed_rf.csv`

## Running the Models

### Naive Bayes Classification

```bash
python main_bayes.py
```

Trains and evaluates:
1. **KDE-based Naive Bayes** (multiclass and binary)
2. **Gaussian-based Naive Bayes** (multiclass)

Uses stratified 5-fold cross-validation with:
- Per-class precision, recall, F1-score
- Confusion matrices
- Classification reports

### Random Forest Classification

```bash
python main_random_forest.py
```

Trains with GridSearchCV to find optimal hyperparameters:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]

Evaluates on:
1. **Multiclass classification** (3 classes: No Disorder, Insomnia, Sleep Apnea)
2. **Binary classification** (Disorder vs No Disorder)

Uses stratified 5-fold cross-validation with:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Top 8 feature importances

## Key Classes and Functions

### Naive Bayes Models (`bayes_model/classifier.py`)

**BayesClassifier**
- KDE-based Naive Bayes with Laplace smoothing
- Handles both numerical (KDE) and categorical (probability table) features
- `fit(data, target_column)`: Train on data
- `predict(data)`: Classify samples

**BayesGaussianClassifier**
- Gaussian distribution assumption for numerical features
- Laplace smoothing for categorical features
- Same interface as BayesClassifier

### Sklearn Wrapper (`bayes_model/sklearn_wrapper.py`)

**BayesSklearnWrapper**
- Makes custom Bayes classifiers compatible with scikit-learn API
- Supports both KDE and Gaussian variants
- Works with cross_validate, GridSearchCV, etc.

### Feature Selection (`bayes_model/feature_selection.py`)

**find_optimal_features()**
- Tests correlation thresholds (> 0.75)
- Uses ShuffleSplit cross-validation (25 splits, 70/30)
- Returns best feature set by F1-score
- Saves to `columns_to_drop.csv`

### Random Forest Pipeline (`random_forest_model/pipeline.py`)

**create_preprocessor()**
- One-hot encodes categorical features
- Passes through numeric features unchanged

**create_pipeline()**
- Combines preprocessor with RandomForestClassifier
- Prevents data leakage in cross-validation

**get_feature_importances()**
- Extracts feature names from encoded categorical columns
- Returns importance scores and sorted indices

### Evaluation Utilities (`common_utils/evaluation_utils.py`)

- `plot_class_distribution()`: Class balance visualization
- `plot_correlation_matrix()`: Feature correlation heatmap
- `print_metrics()`: Classification metrics summary
- `plot_confusion_matrix()`: Confusion matrix visualization
- `print_feature_importances()`: Top N features
- `plot_feature_importances()`: Feature importance bar plot

## Configuration (`common_utils/config.py`)

```python
CATEGORICAL_FEATURES = ["Gender", "Occupation", "BMI"]
NUMERIC_FEATURES = [
    "Age", "Sleep Duration", "Quality of Sleep",
    "Physical Activity Level", "Stress Level",
    "Diastolic Pressure", "Heart Rate", "Daily Steps"
]
TARGET_COLUMN = "Sleep Disorder"
```

## Output Files

Each training run generates:

**Bayes Models**:
- `sleep_processed_bayes.csv` - Preprocessed dataset
- Confusion matrix plots

**Random Forest**:
- `sleep_processed_rf.csv` - Preprocessed dataset
- `confusion_matrix.png` - Multiclass confusion matrix
- `confusion_matrix_binary.png` - Binary confusion matrix
- `feature_importance.png` - Multiclass feature importances
- `feature_importance_binary.png` - Binary feature importances
- `class_distribution.png` - Class balance visualization

## Usage Examples

### Custom Bayes Classifier Training

```python
from bayes_model.classifier import BayesClassifier
from common_utils.data_processing import load_dataset, preprocess_data_basic

df = load_dataset("sleep_quality.csv")
df = preprocess_data_basic(df)

clf = BayesClassifier()
clf.fit(df, "Sleep Disorder")
predictions = clf.predict(df.drop(columns=["Sleep Disorder"]))
```

### Custom Random Forest with Cross-Validation

```python
from sklearn.model_selection import cross_validate
from random_forest_model.pipeline import create_preprocessor, create_pipeline
from common_utils.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

preprocessor = create_preprocessor()
pipeline = create_pipeline(preprocessor)

X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y = df["Sleep Disorder"]

cv_results = cross_validate(pipeline, X, y, cv=5, scoring="f1_weighted")
```

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

See `requirements.txt` for version specifications.

## Notes

- Both preprocessing methods are kept separate to preserve model-specific requirements
- Categorical and numerical features are defined in a single config for consistency
- Evaluation functions are centralized but each model script handles its own output
- All models use stratified cross-validation to maintain class proportions
