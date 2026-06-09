import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# Naive Bayes classifier using kernel density estimation for continuous features
class BayesClassificator:
    def __init__(self):
        self.cls = None  # Name of the target class column
        self.classesOfAbstraction = None  # Unique class labels
        self.totalSamples = None  # Total number of training samples
        self.kdemodels = {}  # Kernel density estimators for numerical features per class
        self.categorical = {}  # Probability distributions for categorical features per class
        self.classLength = {}  # Count of samples per class
        self.uniqueVal = {}  # Number of unique values per categorical feature

    # Train the classifier on labeled data
    def fit(self, data, classes):
        self.totalSamples = len(data)
        self.cls = classes
        self.classesOfAbstraction = data[classes].unique()
        self.classLength = data[classes].value_counts().to_dict()

        # Store categorical feature probabilities for each class using Laplace smoothing
        grouped = data.groupby(classes)
        for name, group in grouped:
            features = group.drop(columns=[classes]).select_dtypes(exclude=['number'])
            temp_dict_cat = {}
            temp_dict_unique = {}
            for col in features.columns:
                temp_dict_unique[col] = data[col].nunique()
                # Laplace smoothing adds 1 to counts to prevent zero probabilities
                temp_dict_cat[col] = (features[col].value_counts()+1)/(self.classLength[name]+temp_dict_unique[col])

            self.categorical[name] = temp_dict_cat
            self.uniqueVal[name] = temp_dict_unique

        # Fit kernel density estimators for numerical features
        for name, group in grouped:
            features = group.drop(columns=[classes]).select_dtypes(include=['number'])
            self.kdemodels[name] = {col: gaussian_kde(features[col]) for col in features.columns}

    # Calculate log probability of a sample belonging to a class
    def bayes_probability(self, x, cls):
        # Sum log probabilities of numerical features using KDE
        x_numerical_proba = sum(np.log(self.kdemodels[cls][col].evaluate([val])[0]) for col, val in x.items() if isinstance(val, (int, float, complex)))

        # Sum log probabilities of categorical features
        class_length = self.classLength[cls]
        x_categorical_proba = sum(np.log(self.categorical[cls][col].get(val, 1/(class_length + self.uniqueVal[cls][col]))) for col, val in x.items() if isinstance(val, str))

        # Combine with log prior probability of the class
        probability = x_numerical_proba + x_categorical_proba + np.log(self.classLength[cls]/self.totalSamples)
        return probability

    # Classify samples by selecting class with highest posterior probability
    def predict(self, data):
        data_unclassified = data.drop(columns=[self.cls], errors='ignore')
        data_predictions = data_unclassified.copy()

        # Calculate probability for each class
        for cls in self.classesOfAbstraction:
            data_predictions[cls] = data_unclassified.apply(lambda x: self.bayes_probability(x, cls), axis = 1)

        # Select class with maximum probability and drop probability columns
        data_classified = data_predictions
        data_classified[self.cls] = data_classified.loc[:, self.classesOfAbstraction].idxmax(axis=1)
        data_classified = data_classified.drop(columns = self.classesOfAbstraction)
        return data_classified


# Naive Bayes classifier using Gaussian distribution for continuous features
class BayesGuassianClassificator:
    def __init__(self):
        self.cls = None  # Name of the target class column
        self.totalSamples = None  # Total number of training samples
        self.statistics = {}  # Mean and standard deviation per numerical feature per class
        self.classLength = {}  # Count of samples per class
        self.categorical = {}  # Probability distributions for categorical features per class
        self.uniqueVal = {}  # Number of unique values per categorical feature

    # Train the classifier on labeled data
    def fit(self, data, classes):
        self.totalSamples = len(data)
        self.cls = classes
        self.classesOfAbstraction = data[classes].unique()
        self.classLength = data[classes].value_counts().to_dict()

        # Store categorical feature probabilities for each class using Laplace smoothing
        grouped = data.groupby(classes)
        for name, group in grouped:
            features = group.drop(columns=[classes]).select_dtypes(exclude=['number'])
            temp_dict_cat = {}
            temp_dict_unique = {}
            for col in features.columns:
                temp_dict_unique[col] = data[col].nunique()
                # Laplace smoothing adds 1 to counts to prevent zero probabilities
                temp_dict_cat[col] = (features[col].value_counts()+1)/(self.classLength[name]+temp_dict_unique[col])

            self.categorical[name] = temp_dict_cat
            self.uniqueVal[name] = temp_dict_unique

        # Calculate mean and standard deviation for numerical features per class
        for name, group in grouped:
            # Add small epsilon to std to avoid division by zero
            self.statistics[name] = pd.DataFrame([group.mean(numeric_only=True), group.std(numeric_only=True) + 1e-9], index = ['mean', 'std'])

    # Calculate log probability using Gaussian distribution assumption
    def GaussDistributionDensity(self, x, cls):
        mean = self.statistics[cls].loc['mean']
        std = self.statistics[cls].loc['std']
        class_length = self.classLength[cls]

        # Log probability density for numerical features using Gaussian formula
        x_numerical_proba = sum(-np.log(std[col] * np.sqrt(2 * np.pi)) + (-pow(val - mean[col],2)/(2 * pow(std[col], 2))) for col, val in x.items() if isinstance(val, (int, float, complex)))

        # Log probability for categorical features
        x_categorical_proba = sum(np.log(self.categorical[cls][col].get(val, 1/(class_length + self.uniqueVal[cls][col]))) for col, val in x.items() if isinstance(val, str))

        # Combine with log prior probability of the class
        probability = x_numerical_proba + x_categorical_proba + np.log(class_length/self.totalSamples)
        return probability

    # Classify samples by selecting class with highest posterior probability
    def predict(self, unclfData):
        if self.cls in unclfData.columns:
            unclfData = unclfData.drop(columns = [self.cls], errors='ignore')
        clfData = unclfData.copy()

        # Calculate probability for each class
        for cls in self.statistics:
            log_prior = np.log(self.classLength[cls] / self.totalSamples)
            clfData[cls] = unclfData.apply(
                lambda x: self.GaussDistributionDensity(x, cls).sum() + log_prior,
                axis=1
            )

        # Select class with maximum probability and drop probability columns
        clfData[self.cls] = clfData.loc[:, self.statistics.keys()].idxmax(axis=1)
        clfData = clfData.drop(columns=self.statistics.keys())
        return clfData