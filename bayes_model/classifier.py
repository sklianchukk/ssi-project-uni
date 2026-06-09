import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


class BayesClassifier:
    """Naive Bayes classifier using kernel density estimation for continuous features."""

    def __init__(self):
        self.target_column = None
        self.classes = None
        self.total_samples = None
        self.kde_models = {}
        self.categorical_probs = {}
        self.class_counts = {}
        self.unique_values = {}

    def fit(self, data: pd.DataFrame, target_column: str) -> None:
        """Train the classifier on labeled data."""
        self.total_samples = len(data)
        self.target_column = target_column
        self.classes = data[target_column].unique()
        self.class_counts = data[target_column].value_counts().to_dict()

        grouped = data.groupby(target_column)

        # Store categorical feature probabilities with Laplace smoothing
        for class_label, group in grouped:
            categorical_features = group.drop(columns=[target_column]).select_dtypes(
                exclude=["number"]
            )
            cat_probs = {}
            unique_vals = {}

            for col in categorical_features.columns:
                unique_vals[col] = data[col].nunique()
                # Laplace smoothing: add 1 to counts
                cat_probs[col] = (
                    categorical_features[col].value_counts() + 1
                ) / (self.class_counts[class_label] + unique_vals[col])

            self.categorical_probs[class_label] = cat_probs
            self.unique_values[class_label] = unique_vals

        # Fit kernel density estimators for numerical features
        for class_label, group in grouped:
            numerical_features = group.drop(columns=[target_column]).select_dtypes(
                include=["number"]
            )
            self.kde_models[class_label] = {
                col: gaussian_kde(numerical_features[col]) for col in numerical_features.columns
            }

    def _calculate_probability(self, sample: pd.Series, class_label: str) -> float:
        """Calculate log probability of sample belonging to class."""
        # Log probability from numerical features
        log_prob_numerical = sum(
            np.log(self.kde_models[class_label][col].evaluate([val])[0])
            for col, val in sample.items()
            if isinstance(val, (int, float, complex))
        )

        # Log probability from categorical features
        class_count = self.class_counts[class_label]
        log_prob_categorical = sum(
            np.log(
                self.categorical_probs[class_label][col].get(
                    val, 1 / (class_count + self.unique_values[class_label][col])
                )
            )
            for col, val in sample.items()
            if isinstance(val, str)
        )

        # Add log prior probability
        log_prior = np.log(class_count / self.total_samples)

        return log_prob_numerical + log_prob_categorical + log_prior

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify samples by selecting class with highest posterior probability."""
        data_clean = data.drop(columns=[self.target_column], errors="ignore")
        predictions = data_clean.copy()

        # Calculate probability for each class
        for class_label in self.classes:
            predictions[class_label] = data_clean.apply(
                lambda x: self._calculate_probability(x, class_label), axis=1
            )

        # Select class with maximum probability
        predictions[self.target_column] = predictions.loc[:, self.classes].idxmax(axis=1)
        predictions = predictions.drop(columns=self.classes)

        return predictions


class BayesGaussianClassifier:
    """Naive Bayes classifier using Gaussian distribution for continuous features."""

    def __init__(self):
        self.target_column = None
        self.classes = None
        self.total_samples = None
        self.statistics = {}
        self.class_counts = {}
        self.categorical_probs = {}
        self.unique_values = {}

    def fit(self, data: pd.DataFrame, target_column: str) -> None:
        """Train the classifier on labeled data."""
        self.total_samples = len(data)
        self.target_column = target_column
        self.classes = data[target_column].unique()
        self.class_counts = data[target_column].value_counts().to_dict()

        grouped = data.groupby(target_column)

        # Store categorical feature probabilities with Laplace smoothing
        for class_label, group in grouped:
            categorical_features = group.drop(columns=[target_column]).select_dtypes(
                exclude=["number"]
            )
            cat_probs = {}
            unique_vals = {}

            for col in categorical_features.columns:
                unique_vals[col] = data[col].nunique()
                cat_probs[col] = (
                    categorical_features[col].value_counts() + 1
                ) / (self.class_counts[class_label] + unique_vals[col])

            self.categorical_probs[class_label] = cat_probs
            self.unique_values[class_label] = unique_vals

        # Calculate mean and std for numerical features
        for class_label, group in grouped:
            # Add small epsilon to std to avoid division by zero
            self.statistics[class_label] = pd.DataFrame(
                [
                    group.mean(numeric_only=True),
                    group.std(numeric_only=True) + 1e-9,
                ],
                index=["mean", "std"],
            )

    def _gaussian_probability(self, sample: pd.Series, class_label: str) -> float:
        """Calculate log probability using Gaussian distribution."""
        mean = self.statistics[class_label].loc["mean"]
        std = self.statistics[class_label].loc["std"]
        class_count = self.class_counts[class_label]

        # Log probability from numerical features
        log_prob_numerical = sum(
            -np.log(std[col] * np.sqrt(2 * np.pi))
            + (-((val - mean[col]) ** 2) / (2 * std[col] ** 2))
            for col, val in sample.items()
            if isinstance(val, (int, float, complex))
        )

        # Log probability from categorical features
        log_prob_categorical = sum(
            np.log(
                self.categorical_probs[class_label][col].get(
                    val, 1 / (class_count + self.unique_values[class_label][col])
                )
            )
            for col, val in sample.items()
            if isinstance(val, str)
        )

        # Add log prior probability
        log_prior = np.log(class_count / self.total_samples)

        return log_prob_numerical + log_prob_categorical + log_prior

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify samples by selecting class with highest posterior probability."""
        data_clean = data.drop(columns=[self.target_column], errors="ignore")
        predictions = data_clean.copy()

        # Calculate probability for each class
        for class_label in self.statistics.keys():
            predictions[class_label] = data_clean.apply(
                lambda x: self._gaussian_probability(x, class_label), axis=1
            )

        # Select class with maximum probability
        predictions[self.target_column] = predictions.loc[:, self.statistics.keys()].idxmax(
            axis=1
        )
        predictions = predictions.drop(columns=self.statistics.keys())

        return predictions
