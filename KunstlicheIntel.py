import pandas as pd
import numpy as np

class BayesClassificator:
    def __init__(self):
        self.cls = None
        self.totalSamples = None 
        self.statistics = {}
        self.classLength = {}

    def fit(self, data, classes):
        grouped = data.groupby(classes)
        self.totalSamples = len(data)
        self.cls = classes
        for name, group in grouped:
            self.classLength[name] = len(group)
            features = group.drop(columns=[classes])
            self.statistics[name] = pd.DataFrame([features.mean(numeric_only=True), features.std(numeric_only=True) + 1e-9], index = ['mean', 'std'])
    
    # Główna funkcja do predykcji obliczająca gęstość poszczególnych atrybutów elementów w klasie
    #
    #

    def predict(self, unclfData):
        unclfData = unclfData.copy()
        
        if self.cls in unclfData.columns:
            features = unclfData.drop(columns=[self.cls])
        else:
            features = unclfData
            
        for cls in self.statistics:
            log_prior = np.log(self.classLength[cls] / self.totalSamples)
            unclfData[cls] = features.apply(
                lambda x: self.TriangleDistribution(x, cls).sum() + log_prior, 
                axis=1
            )
        
        unclfData[self.cls] = unclfData.loc[:, self.statistics.keys()].idxmax(axis=1)
        unclfData = unclfData.drop(columns=self.statistics.keys())
        return unclfData


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