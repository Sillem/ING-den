### MODULE PREPARED FOR IMPORTING PIPELINE FOR DATA TRANSFORMATION IN DIFFERENT MODULES OF THE PROJECTS

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

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

# Zakłada że moduły będą uruchamiane z roota repo
names_xlsx = pd.read_excel('./variables_description.xlsx')
#Słownik zmian nazw kolumn
names = {f"{names_xlsx['Column'][i]}":f"{names_xlsx['Description'][i]}" for i in range(5, len(names_xlsx))}

subtypes_list = [discrete_variables, continuous_variables, 
binary_variables, categorical_nominal_variables, datetime_variables]

for subtype_idx in range(len(subtypes_list)):
    for variable_idx in range(len(subtypes_list[subtype_idx])):
        if subtypes_list[subtype_idx][variable_idx] in names.keys():
            subtypes_list[subtype_idx][variable_idx] = names[subtypes_list[subtype_idx][variable_idx]]

zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
vars_for_zero_impute = ['Application data: income of second applicant', 'Application data: profession of second applicant', 'Value of the goods (car)']

add_category_imputer = SimpleImputer(strategy="constant", fill_value=2)
vars_for_add_category_impute = ['Property ownership for property renovation', 'Clasification of the vehicle (Car, Motorbike)']
mode_imputer = SimpleImputer(strategy="most_frequent")

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

impute_column_transformer = ColumnTransformer([
    ("zero_fill", zero_imputer, vars_for_zero_impute),
    ("add_third_category", add_category_imputer, vars_for_add_category_impute),
    ("mode_impute", mode_imputer, vars_for_mode_impute),
    ("fill_zeros_but_add_var", SimpleImputeAddFeature(vars_for_fill_zeros_but_add_var), vars_for_fill_zeros_but_add_var)],
    remainder="passthrough"
)

def remove_nans_pipeline(X : pd.DataFrame) -> pd.DataFrame:
    # takes in the raw dataframe that comes with the task
    X = X.rename(columns=names)
    transformed_X = impute_column_transformer.fit_transform(X)
    
    transformed_X_df = pd.DataFrame(transformed_X, columns=impute_column_transformer.get_feature_names_out())
    
    ft_remove_spendings_estimations = FunctionTransformer(lambda df: df[df['remainder__Spendings estimation'].notna()])
    return ft_remove_spendings_estimations.transform(transformed_X_df)