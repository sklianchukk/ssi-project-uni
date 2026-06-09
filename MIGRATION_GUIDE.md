# Migration Guide: Old Files to New Structure

This document maps old files to their new locations in the refactored project.

## File Mapping

| Old File | New Location | Notes |
|----------|--------------|-------|
| `KunstlicheIntel.py` | `bayes_model/classifier.py` | Renamed classes: BayesClassificator → BayesClassifier, BayesGuassianClassificator → BayesGaussianClassifier. Improved formatting and documentation. |
| `Bayes_sklearn.py` | `bayes_model/sklearn_wrapper.py` | Enhanced with classifier_type parameter to support both KDE and Gaussian variants. |
| `config.py` | `common_utils/config.py` | Added TARGET_COLUMN constant. |
| `data_processing.py` (Random Forest version) | `common_utils/data_processing.py` | Renamed to preprocess_data_for_random_forest(). Merged with basic preprocessing. |
| `data_processing_random_forest.py` | `common_utils/data_processing.py` | Merged as preprocess_data_basic(). |
| `detectThreshold.py` | `bayes_model/feature_selection.py` | Refactored into find_optimal_features() function with improved documentation. |
| `evaluation.py` (Random Forest) | `random_forest_model/pipeline.py` + `common_utils/evaluation_utils.py` | Pipeline and model creation in pipeline.py. Visualization and metrics in evaluation_utils.py. |
| `model.py` | `random_forest_model/pipeline.py` | Kept as-is, provides create_preprocessor(), create_pipeline(), get_feature_importances(). |
| `train_predict.py` | `main_bayes.py` | Refactored into modular evaluate_classifier() function. Cleaner output formatting. |
| `main.py` | `main_random_forest.py` | Refactored into train_and_evaluate() function. Supports both multiclass and binary. |

## Key Improvements

### Code Organization
- **Separation of Concerns**: Model-specific code in dedicated directories
- **Common Utilities**: Shared functions extracted to common_utils/
- **Clear Entry Points**: Two main scripts (main_bayes.py, main_random_forest.py)

### Code Quality
- **Renamed Classes**: BayesClassificator → BayesClassifier (correct spelling)
- **Better Documentation**: Docstrings and comments added throughout
- **Consistent APIs**: Both classifiers follow sklearn conventions
- **Modular Functions**: Evaluation code refactored into reusable functions

### Functionality Preserved
- ✅ KDE-based Naive Bayes classification
- ✅ Gaussian-based Naive Bayes classification
- ✅ Random Forest with GridSearchCV
- ✅ Stratified k-fold cross-validation
- ✅ Binary and multiclass support
- ✅ Feature importance analysis
- ✅ Confusion matrices and classification reports
- ✅ Correlation-based feature selection (Bayes)

## Breaking Changes

### Import Statements

**Old**:
```python
import KunstlicheIntel as ki
from Bayes_sklearn import BayesSklearnWrapper
```

**New**:
```python
from bayes_model.classifier import BayesClassifier, BayesGaussianClassifier
from bayes_model.sklearn_wrapper import BayesSklearnWrapper
```

### Class Names

- `BayesClassificator` → `BayesClassifier`
- `BayesGuassianClassificator` → `BayesGaussianClassifier`

### Function Names

- `preprocess_data()` → `preprocess_data_for_random_forest()` (for RF)
- `load_dataset()` signature unchanged

### Data Loading

**Old**:
```python
sleep = pd.read_csv("sleepProcessed.csv")
```

**New** (kept same, but output files are:
- `sleep_processed_bayes.csv` (from main_bayes.py)
- `sleep_processed_rf.csv` (from main_random_forest.py)

## Migration Steps (Optional)

If you want to run old scripts with minimal changes:

1. Update imports to use new module paths
2. Replace classifier references:
   - `ki.BayesClassificator()` → `BayesClassifier()`
   - `ki.BayesGuassianClassificator()` → `BayesGaussianClassifier()`
3. Update config imports: `from config import ...` → `from common_utils.config import ...`
4. Use new data loading: `from common_utils.data_processing import load_dataset, preprocess_data_*`

## Recommended Next Steps

1. **Delete Old Files**: After verifying new structure works
   - `KunstlicheIntel.py`
   - `Bayes_sklearn.py`
   - `data_processing.py` (old version)
   - `data_processing_random_forest.py`
   - `detectThreshold.py`
   - `evaluation.py`
   - `train_predict.py`
   - `main.py`
   - `model.py` (functionality moved to random_forest_model/pipeline.py)

2. **Update Version Control**: Add new structure to git

3. **Run Tests**: Verify both main_bayes.py and main_random_forest.py work correctly

4. **Update Documentation**: Update any project documentation referencing old file locations
