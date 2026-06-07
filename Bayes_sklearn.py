# Scikit-learn interface adapters for estimator functionality
from sklearn.base import BaseEstimator, ClassifierMixin
import KunstlicheIntel as ki
import pandas as pd

# Wrapper class to make custom Bayes classifier compatible with scikit-learn API
class BayesSklearnWrapper(BaseEstimator, ClassifierMixin):
    # Initialize with target column name (the column to predict)
    def __init__(self, target_column='Sleep Disorder' ):
        self.target_column = target_column
        self.model = ki.BayesClassificator()
        self.classes_ = None  # Will store unique class labels after fitting

    # Train the model by combining features and target into a single dataframe
    def fit(self, X, y):
        full_df = pd.concat([X, y], axis=1)
        self.model.fit(full_df, self.target_column)
        self.classes_ = y.unique()
        return self

    # Make predictions on new data and return predicted values
    def predict(self, X):
        prediction_df = self.model.predict(X)
        return prediction_df[self.target_column].values