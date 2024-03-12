### MODULE PREPARED FOR IMPORTING PIPELINE FOR DATA TRANSFORMATION IN DIFFERENT MODULES OF THE PROJECTS

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector
from feature_engine.encoding import WoEEncoder

discrete_variables = ['ID', 'customer_id', 'Var1', 'Var15', 'Var16', 'Var20', 'Var21', 'Var22',
                      	'Var23', 'Var29', 'Var4', 'Var5', 'Var9', 'Var24', 'Var30', 'Var6'
]

continuous_variables = [
    'Var7', 'Var8', 'Var10', 
    'Var17', 'Var25', 'Var26', '_r_'
]

binary_variables = [
    'target', 'Application_status', 'Var18', 
    'Var19', 'Var27', 'Var28'
]

categorical_nominal_variables = [
    'Var2', 'Var3', 'Var11', 'Var12', 'Var14'
]


datetime_variables = [
    'application_date', 'Var13'
]

names_xlsx = pd.read_excel('./variables_description.xlsx')
#Słownik zmian nazw kolumn
names = {f"{names_xlsx['Column'][i]}":f"{names_xlsx['Description'][i]}" for i in range(5, len(names_xlsx))}

def rename_list(lista):
    for idx in range(len(lista)):
        if lista[idx] in names.keys():
            lista[idx] = names[lista[idx]]
    return lista

discrete_variables = rename_list(discrete_variables)
continuous_variables = rename_list(continuous_variables)
binary_variables = rename_list(binary_variables)
categorical_nominal_variables = rename_list(categorical_nominal_variables)
datetime_variables = rename_list(datetime_variables)

def remove_nans(X : pd.DataFrame, columns=['target', 'Spendings estimation']) -> pd.DataFrame:
    """Funkcja do wywalania wierszy które mają NaN w którejś z kolumn podanych w liście.
    

    Args:
        X (pd.DataFrame): dataframe do przetworzenia (usunięcia wierszy). Ten surowy z URLa.
        columns (list, optional): Kolumny z oryginalnego df (opisowe, nie VarX). 
        Z których wiersze z NaNami.
        Defaults to ['target', 'Spendings estimation'].

    Returns:
        pd.DataFrame pd.Series: Dataframe z danymi treningowymi, dataframe z labelkami
    """
    X = X.rename(columns=names)
    X = X.set_index('ID')
    for column in columns:
        X = X[X[column].notna()]
    return X.drop(['target'], axis=1), X['target']

def fix_encodings(X : pd.DataFrame) -> pd.DataFrame:
    """Tutaj sztywno zmieniam zepsute encodingi w danych kolumnach

    Args:
        X (pd.DataFrame): dataframe po użyciu remove_nans

    Returns:
        pd.DataFrame: dataframe z poprawionymi encodingami
    """
    X_copy = X.copy()
    if 'Distribution channel' in X.columns:
        X_copy['Distribution channel'] = X_copy['Distribution channel'].replace("Direct", 1)
        X_copy['Distribution channel'] = X_copy['Distribution channel'].replace("Broker", 2)    
        X_copy['Distribution channel'] = X_copy['Distribution channel'].replace("Online", 3)

    if 'Application_status' in X.columns:
        X_copy['Application_status'] = X_copy['Application_status'].replace("Approved", 1)
        X_copy['Application_status'] = X_copy['Application_status'].replace("Rejected", 0)
    return X_copy

vars_for_zero_impute = ['Application data: income of second applicant', 'Application data: profession of second applicant', 'Value of the goods (car)']
vars_for_add_category_impute = ['Property ownership for property renovation', 'Clasification of the vehicle (Car, Motorbike)']
vars_for_mode_impute = ['Loan purpose', 'Distribution channel']
vars_for_fill_zeros_but_add_var = ["Amount on current account", "Amount on savings account"]

class SimpleImputeAddFeature(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns # Lista kolumn do transformacji

    def fit(self, X, y=None):
        # W fit nic nie musimy robić, ale musi być obecna
        return self

    def transform(self, X):
        # Tworzymy kopię, aby nie modyfikować oryginalnego DataFrame
        X_copy = X.copy()
        
        for column in self.columns:
            # Dodajemy nową kolumnę z wartościami 0 i 1
            X_copy[column + '_was_missing'] = X_copy[column].isnull().astype(int)
            
            # Simple impute - zamieniamy NaN na 0
            X_copy[column] = X_copy[column].fillna(0)
        
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
       if input_features is None:
           input_features = self.columns
       # Zakładając, że self.columns zawiera cechy, które zostały przetworzone
       output_features = np.concatenate([input_features, [f"{col}_was_missing" for col in self.columns]])
       return output_features
    
zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
add_category_imputer = SimpleImputer(strategy="constant", fill_value=2)
mode_imputer = SimpleImputer(strategy="most_frequent")

impute_column_transformer = ColumnTransformer([
    ("zero_fill", zero_imputer, vars_for_zero_impute),
    ("add_third_category", add_category_imputer, vars_for_add_category_impute),
    ("mode_impute", make_pipeline(FunctionTransformer(fix_encodings), mode_imputer), vars_for_mode_impute),
    ("fill_zeros_but_add_var", SimpleImputeAddFeature(vars_for_fill_zeros_but_add_var), vars_for_fill_zeros_but_add_var),
    ("application_status_transform", FunctionTransformer(fix_encodings), ['Application_status'])
    ],
    remainder="passthrough"
).set_output(transform='pandas')

def make_dataframe_numeric_again(X : pd.DataFrame) -> pd.DataFrame:
    X_copy = X.copy()
    for column in X:
        if column.split('__')[1] not in datetime_variables: 
            X_copy[column] = pd.to_numeric(X[column])
    return X_copy

numericTransformer = FunctionTransformer(make_dataframe_numeric_again)

num_regex = "^(.*)("
for num_feature in discrete_variables + continuous_variables:
    num_feature = num_feature.replace(')', '\)').replace('(', '\(')
    num_regex+=num_feature+'|'
num_regex=num_regex[:-1] # removing last |
num_regex+=')$'

#lets build nominal feature regex selector
nominal_regex = "^(.*)("
for cat_feature in categorical_nominal_variables:
        nominal_regex+=cat_feature+'|'
nominal_regex=nominal_regex[:-1]
nominal_regex+=')$'

feature_transform_transformer = ColumnTransformer([
    ("scale", StandardScaler(), make_column_selector(num_regex)),
    ("one_hot_encode", OneHotEncoder(sparse_output=False), make_column_selector(nominal_regex))
],
    remainder="passthrough").set_output(transform="pandas")

feature_transform_transformer_woe = ColumnTransformer([
    ("scale", StandardScaler(), make_column_selector(num_regex)),
    ("woe_encode", WoEEncoder(ignore_format=True), make_column_selector(nominal_regex))
],
    remainder="passthrough").set_output(transform="pandas")

full_pipeline_logisitic = make_pipeline(impute_column_transformer, numericTransformer, feature_transform_transformer_woe)
full_pipeline_ml = make_pipeline(impute_column_transformer, numericTransformer, feature_transform_transformer)

