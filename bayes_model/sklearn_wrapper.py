import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from bayes_model.classifier import BayesClassifier, BayesGaussianClassifier


class BayesSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make custom Bayes classifier compatible with scikit-learn API."""

    def __init__(self, target_column="Sleep Disorder", classifier_type="kde"):
        """
        Args:
            target_column: Name of the target column to predict.
            classifier_type: "kde" for KDE-based or "gaussian" for Gaussian-based.
        """
        self.target_column = target_column
        self.classifier_type = classifier_type
        self.model = (
            BayesClassifier()
            if classifier_type == "kde"
            else BayesGaussianClassifier()
        )
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        full_df = pd.concat([X, y], axis=1)
        self.model.fit(full_df, self.target_column)
        self.classes_ = y.unique()
        return self

    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        prediction_df = self.model.predict(X)
        return prediction_df[self.target_column].values
