import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score
from itertools import combinations

def gini(y_true, y_score):
    return 2*roc_auc_score(y_true, y_score)-1

class GiniSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold # threshold for column 

    def fit(self, X, y=None):
        self.col_to_drop = [ i for i in X.columns if gini(y, X[i]) <= self.threshold]
        return self

    def transform(self, X):
        # Tworzymy kopię, aby nie modyfikować oryginalnego DataFrame
        return X.copy(deep=True).drop(self.col_to_drop ,axis=1)
    
    def get_feature_names_out(self, input_features=None):
       return self.col_to_drop

class OverCorrelatedDropper(GiniSelector):
    def fit(self, X, y=None):
        self.col_to_drop=[]
        CorrelationMatrix = np.abs(X.corr())
        OverCorrelated = [ (i, j, CorrelationMatrix[i][j])
                            for i,j in combinations(CorrelationMatrix.columns,2)
                          if CorrelationMatrix[i][j] > self.threshold]
        OverCorrelatedSet = set([ j for i in OverCorrelated for j in i[:-1]])
        gini_dict= { i: gini(y, X[i]) for i in OverCorrelatedSet}
        while OverCorrelatedSet:
            minimal_gini= min( gini_dict, key=lambda x: gini_dict[x])
            OverCorrelated = [i for i in OverCorrelated if i[0]!=minimal_gini and i[1]!=minimal_gini]
            OverCorrelatedSet = set([j for i in OverCorrelated for j in i[:-1]])
            del gini_dict[minimal_gini]
            self.col_to_drop.append(minimal_gini)
        #self.col_to_drop = [ i for i in X.columns if 2*roc_auc_score(y, X[i])-1 <= self.threshold]
        return self
