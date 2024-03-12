import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score

class GiniSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold # threshold for column 

    def fit(self, X, y=None):
        self.col_to_drop = [ i for i in X.columns if 2*roc_auc_score(y, X[i])-1 <= self.threshold]
        return self

    def transform(self, X):
        # Tworzymy kopię, aby nie modyfikować oryginalnego DataFrame
        return X.copy(deep=True).drop(self.col_to_drop ,axis=1)
    
    def get_feature_names_out(self, input_features=None):
       return self.col_to_drop

