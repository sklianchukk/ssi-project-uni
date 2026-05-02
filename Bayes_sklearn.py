from sklearn.base import BaseEstimator, ClassifierMixin
import KunstlicheIntel as ki
import pandas as pd 

class BayesSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, target_column='Sleep Disorder' ):
        self.target_column = target_column
        self.model = ki.BayesClassificator()
        self.classes_ = None

    def fit(self, X, y):
        full_df = pd.concat([X, y], axis=1)
        self.model.fit(full_df, self.target_column)
        self.classes_ = y.unique()
        return self

    def predict(self, X):
        prediction_df = self.model.predict(X)
        return prediction_df[self.target_column].values