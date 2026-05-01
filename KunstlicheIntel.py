import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

class BayesClassificator:
    def __init__(self):
        self.cls = None
        self.classesOfAbstraction = None
        self.totalSamples = None 
        self.kdemodels = {}
        self.categorical = {}
        self.classLength = {}
        self.uniqueVal = {}

    def fit(self, data, classes):
        
        self.totalSamples = len(data)
        self.cls = classes
        self.classesOfAbstraction = data[classes].unique()
        self.classLength = data[classes].value_counts().to_dict()


        # ==== Indetifying probability of attribute's categories for every class ====
        grouped = data.groupby(classes)
        for name, group in grouped:
            
            features = group.drop(columns=[classes]).select_dtypes(exclude=['number'])
            # Applying Laplace smoothing to handele the problem of zero probabilities
            temp_dict_cat = {}
            temp_dict_unique = {}
            for col in features.columns:
                temp_dict_unique[col] = data[col].nunique()
                temp_dict_cat[col] = (features[col].value_counts()+1)/(self.classLength[name]+temp_dict_unique[col]) 
            
            self.categorical[name] = temp_dict_cat
            self.uniqueVal[name] = temp_dict_unique

        # ==== Indetifying kernel density estimator objects for numerical columns for every class ====
        for name, group in grouped:
            features = group.drop(columns=[classes]).select_dtypes(include=['number'])
            self.kdemodels[name] = {col: gaussian_kde(features[col]) for col in features.columns}
        
    
    # Główna funkcja do predykcji obliczająca gęstość poszczególnych atrybutów elementów w klasie

    def bayes_probability(self, x, cls):
        x_numerical_proba = sum(np.log(self.kdemodels[cls][col].evaluate([val])[0]) for col, val in x.items() if isinstance(val, (int, float, complex)))

        class_length = self.classLength[cls]
        x_categorical_proba = sum(np.log(self.categorical[cls][col].get(val, 1/(class_length + self.uniqueVal[cls][col]))) for col, val in x.items() if isinstance(val, str))
        probability = x_numerical_proba + x_categorical_proba + np.log(self.classLength[cls]/self.totalSamples)
        return probability

    
    def predict(self, data):
        data_unclassified = data.drop(columns=[self.cls])
        data_predictions = data_unclassified.copy()
        
        for cls in self.classesOfAbstraction:
            data_predictions[cls] = data_unclassified.apply(lambda x: self.bayes_probability(x, cls), axis = 1)
        
        data_classified = data_predictions
        data_classified[self.cls] = data_classified.loc[:, self.classesOfAbstraction].idxmax(axis=1)
        data_classified = data_classified.drop(columns = self.classesOfAbstraction)
        return data_classified


    def GaussDistributionDensity(self, x, cls):
        mean = self.statistics[cls].loc['mean']
        std = self.statistics[cls].loc['std']
        log_prefactor = -np.log(std * np.sqrt(2 * np.pi))
        exponent_part = -pow(x - mean, 2) / (2 * pow(std, 2))
        return log_prefactor + exponent_part

    # def predict(self, unclfData):
    #     unclfData = unclfData.copy()
        
    #     if self.cls in unclfData.columns:
    #         features = unclfData.drop(columns=[self.cls])
    #     else:
    #         features = unclfData
            
    #     for cls in self.statistics:
    #         log_prior = np.log(self.classLength[cls] / self.totalSamples)
    #         unclfData[cls] = features.apply(
    #             lambda x: self.GaussDistributionDensity(x, cls).sum() + log_prior, 
    #             axis=1
    #         )
        
    #     unclfData[self.cls] = unclfData.loc[:, self.statistics.keys()].idxmax(axis=1)
    #     unclfData = unclfData.drop(columns=self.statistics.keys())
    #     return unclfData

class BayesGuassianClassificator:
    def __init__(self):
        self.cls = None
        self.totalSamples = None 
        self.statistics = {}
        self.classLength = {}
        self.categorical = {}
        self.uniqueVal = {}

    def fit(self, data, classes):
        
        self.totalSamples = len(data)
        self.cls = classes
        self.classesOfAbstraction = data[classes].unique()
        self.classLength = data[classes].value_counts().to_dict()


        # ==== Indetifying probability of attribute's categories for every class ====
        grouped = data.groupby(classes)
        for name, group in grouped:
            
            features = group.drop(columns=[classes]).select_dtypes(exclude=['number'])
            # Applying Laplace smoothing to handele the problem of zero probabilities
            temp_dict_cat = {}
            temp_dict_unique = {}
            for col in features.columns:
                temp_dict_unique[col] = data[col].nunique()
                temp_dict_cat[col] = (features[col].value_counts()+1)/(self.classLength[name]+temp_dict_unique[col]) 
            
            self.categorical[name] = temp_dict_cat
            self.uniqueVal[name] = temp_dict_unique

        # ==== Indetifying mean and standard deviation for numerical columns for every class ====
        for name, group in grouped:
            self.statistics[name] = pd.DataFrame([group.mean(numeric_only=True), group.std(numeric_only=True) + 1e-9], index = ['mean', 'std'])

    def GaussDistributionDensity(self, x, cls):
        mean = self.statistics[cls].loc['mean']
        std = self.statistics[cls].loc['std']
        # log_prefactor = -np.log(std * np.sqrt(2 * np.pi))
        # exponent_part = -pow(x - mean, 2) / (2 * pow(std, 2))
        class_length = self.classLength[cls]
        x_numerical_proba = sum(-np.log(std[col] * np.sqrt(2 * np.pi)) + (-pow(val - mean[col],2)/(2 * pow(std[col], 2))) for col, val in x.items() if isinstance(val, (int, float, complex)))
        x_categorical_proba = sum(np.log(self.categorical[cls][col].get(val, 1/(class_length + self.uniqueVal[cls][col]))) for col, val in x.items() if isinstance(val, str))
        probability = x_numerical_proba + x_categorical_proba + np.log(class_length/self.totalSamples)
        return probability


    def predict(self, unclfData):
        if self.cls in unclfData.columns:
            unclfData = unclfData.drop(columns = [self.cls])
        clfData = unclfData.copy()
        for cls in self.statistics:
            log_prior = np.log(self.classLength[cls] / self.totalSamples)
            clfData[cls] = unclfData.apply(
                lambda x: self.GaussDistributionDensity(x, cls).sum() + log_prior, 
                axis=1
            )
        
        clfData[self.cls] = clfData.loc[:, self.statistics.keys()].idxmax(axis=1)
        clfData = clfData.drop(columns=self.statistics.keys())
        return clfData

class DataProcessing:
    @staticmethod
    def shuffle(x):
        return x.sample(frac=1).reset_index(drop=True)

    @staticmethod
    def normalize(train_data, test_data, columns):
        train_copy = train_data.copy()
        test_copy = test_data.copy()

        t_min = train_copy[columns].min()
        t_max = train_copy[columns].max()
        
        denominator = t_max - t_min
        denominator[denominator == 0] = 1e-9 
        
        train_copy[columns] = (train_copy[columns] - t_min) / denominator
        test_copy[columns] = (test_copy[columns] - t_min) / denominator
        
        return train_copy, test_copy

    @staticmethod
    def split_train_test(x, train_precent, group=None):
        if group != None:
            train_list = []
            test_list = []
            grouped = x.groupby(group)
            
            for name, g in grouped:
                ind = int(train_precent * len(g))
                train_list.append(g.iloc[:ind])
                test_list.append(g.iloc[ind:])

            train_df = pd.concat(train_list, ignore_index=True)
            test_df = pd.concat(test_list, ignore_index=True)
            return train_df, test_df
        else:
            ind = int(train_precent * len(x))
            return x.iloc[:ind], x.iloc[ind:]
        
class Stats:
    @staticmethod
    def accuracy(pred, test, columnPred):
        accuracy_fraction = (pred[columnPred] == test[columnPred]).mean()
        return accuracy_fraction * 100

    @staticmethod
    def precision(pred, test, columnPred):
        results = {}
        classes = test[columnPred].unique()
        for cl in classes:
            TP = len(pred[(pred[columnPred] == test[columnPred]) & (pred[columnPred] == cl)])
            FP = len(pred[(pred[columnPred] != test[columnPred]) & (pred[columnPred] == cl)])
            results[cl] = TP/(TP+FP)
        return results
    
    @staticmethod
    def recall(pred, test, columnPred):
        results = {}
        classes = test[columnPred].unique()
        for cl in classes:
            TP = len(pred[(pred[columnPred] == test[columnPred]) & (pred[columnPred] == cl)])
            FN = len(pred[(pred[columnPred] != cl) & (test[columnPred] == cl)])
            results[cl] = TP/(TP+FN)
        return results
    
    