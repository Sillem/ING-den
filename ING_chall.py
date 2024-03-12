import requests
from io import StringIO
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from pygam import LogisticGAM, s, f
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
import optuna
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import jupyter

train_url = "https://files.challengerocket.com/files/lions-den-ing-2024/development_sample.csv"
test_url = "https://files.challengerocket.com/files/lions-den-ing-2024/testing_sample.csv"


def load_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = response.text
        csv_file = StringIO(csv_data)
        return pd.read_csv(csv_file)
    else:
        print("Nie udało się pobrać danych.")


train = load_data(train_url)
test = load_data(test_url)

train.dropna(subset=['target'], inplace=True)
test.dropna(subset=['target'], inplace=True)

train_id = train["ID"]
test_id = test["ID"]

data = [train, test]
columns_to_drop = ["ID", "customer_id", "application_date", "Application_status", "Var13"]

for frame in data:
    frame.drop(columns=columns_to_drop, inplace=True)

print("Dane treningowe:")
print(train.head())

y_train = train["target"]
y_test = test["target"]

print(y_train)
print(y_train.shape)
print(train.shape)

for frame in data:
    frame.drop(columns=["target"], inplace=True)

print("Dane treningowe:")
print(train.head())
print("\nDane testowe:")
print(test.head())

categorical_features = []
numerical_features = []

for column in train.columns:
    if train[column].dtype == 'object':
        categorical_features.append(column)
    else:
        numerical_features.append(column)

print("categorical: ", categorical_features)
print("numerical: ", numerical_features)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


num_pipeline = Pipeline([
    ("selector", DataFrameSelector(numerical_features)),
    ("imputer", SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(categorical_features)),
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

X_train_transformed = full_pipeline.fit_transform(train)
X_test_transformed = full_pipeline.transform(test)

transformed_cat_columns = full_pipeline.transformer_list[1][1]['encoder'].get_feature_names_out(categorical_features)

transformed_column_names = numerical_features + list(transformed_cat_columns)

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_column_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_column_names)

print("Transformed Train Data:")
print(X_train_transformed_df.head())

print("Transformed Test Data:")
print(X_test_transformed_df.head())

print(X_train_transformed_df.columns)

X = X_train_transformed_df
y = y_train

print(X.shape)
print(y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',  # Use error metric for accuracy
        'tree_method': 'hist',
        'device': 'cuda',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.02, 0.05, 0.1, 0.2]),
        'n_estimators': 1000,
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }

    bst = xgb.train(param, dtrain, evals=[(dvalid, 'eval')], early_stopping_rounds=100, verbose_eval=False)
    preds = bst.predict(dvalid)
    y_pred = [1 if p > 0.5 else 0 for p in preds]
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=2)

#best_params = study.best_params


#XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB XGB

print("XGBoost Model:")
final_xgb_model = xgb.XGBClassifier(objective='binary:logistic')
final_xgb_model.fit(X, y)
y_pred_test_xgb = final_xgb_model.predict_proba(X_test_transformed_df)[:, 1]
print(y_pred_test_xgb)
y_pred_xgb = np.array([1 if p > 0.97 else 0 for p in y_pred_test_xgb])
print("y_pred: ", y_pred_xgb)
print(np.unique(y_pred_xgb, return_counts=True))
# Calculate accuracy
accuracy_test_xgb = accuracy_score(y_test, y_pred_xgb)
print("Accuracy on Test Data:", accuracy_test_xgb)

# Calculate AUC
y_pred_proba_xgb = final_xgb_model.predict_proba(X_test_transformed_df)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print("AUC:", auc_xgb)

# Calculate precision, recall, and F1-score
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1_xgb)

# EBM Model
print("\nExplainable Boosting Machine (EBM) Model:")
ebm = ExplainableBoostingClassifier()
ebm.fit(X, y)

# Calculate auc
y_pred_proba_ebm = ebm.predict_proba(X_test_transformed_df)[:, 1]
print(y_pred_proba_ebm)
auc_ebm = roc_auc_score(y_test, y_pred_proba_ebm)
print("AUC: {:.3f}".format(auc_ebm))

# Calculate accuracy
y_pred_ebm = [1 if p > 0.97 else 0 for p in y_pred_proba_ebm]
accuracy_ebm = accuracy_score(y_test, y_pred_ebm)
print("Accuracy:", accuracy_ebm)

# Calculate precision, recall, and F1-score
precision_ebm = precision_score(y_test, y_pred_ebm)
recall_ebm = recall_score(y_test, y_pred_ebm)
f1_ebm = f1_score(y_test, y_pred_ebm)
print("Precision:", precision_ebm)
print("Recall:", recall_ebm)
print("F1-score:", f1_ebm)

# Logistic Regression Model
print("\nLogistic Regression Model:")
logistic_model = LogisticRegression(max_iter=5000)
logistic_model.fit(X, y)

# Calculate AUC
y_pred_proba_logistic = logistic_model.predict_proba(X_test_transformed_df)[:, 1]
auc_logistic = roc_auc_score(y_test, y_pred_proba_logistic)
print("AUC Score:", auc_logistic)

# Calculate accuracy
y_pred_logistic = logistic_model.predict(X_test_transformed_df)
print("loguogistic:", y_pred_logistic)
y_pred_log = [1 if p > 0.97 else 0 for p in y_pred_logistic]
accuracy_logistic = accuracy_score(y_test, y_pred_log)
print("Accuracy:", accuracy_logistic)

# Calculate precision, recall, and F1-score
precision_logistic = precision_score(y_test, y_pred_log)
recall_logistic = recall_score(y_test, y_pred_log)
f1_logistic = f1_score(y_test, y_pred_log)
print("Precision:", precision_logistic)
print("Recall:", recall_logistic)
print("F1-score:", f1_logistic)

# GAM Model
print("\nGeneralized Additive Model (GAM) Model:")
gam_model = LogisticGAM(s(0) + s(1) + s(2) +s(3) + s(4) + s(5)+ s(6) + s(7) + s(8)+s(9) + s(10) + s(11)+s(12) + s(13) + s(14)+s(15) + s(16) + s(17)+s(18) + s(19) + s(20)+s(21) + s(22) + s(23)+s(24) + s(25) + s(26)+s(27) + s(28) + s(29)+s(30) + s(31) + s(32) +s(33)).fit(X, y)
gam = LogisticGam(reduce(lambda x, y: s(x) + s(y), [i for i in range(34)]).fit(X,y)
# Plot partial dependence for each feature
#for i, term in enumerate(gam_model.terms):
    #if term.isintercept:
        #continue
    #XX = gam_model.generate_X_grid(term=i)
    #pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=.95)

    # Plot partial dependence
    #plt.figure()
    #plt.plot(XX[:, term.feature], pdep)
    #plt.title(f'Partial Dependence for Term {i}')
    #plt.xlabel(f'Feature {term.feature}')
    #plt.ylabel('Partial Dependence')
    #plt.show()

# Predict probabilities on the test set
y_pred_proba_gam = gam_model.predict_proba(X_test_transformed_df.values)
print("gam:", y_pred_proba_gam)
# Calculate AUC
auc_gam = roc_auc_score(y_test, y_pred_proba_gam)
print("AUC:", auc_gam)

# Convert predicted probabilities to binary predictions
y_pred_gam = [1 if p > 0.97 else 0 for p in y_pred_proba_gam]

# Calculate accuracy
accuracy_gam = accuracy_score(y_test, y_pred_gam)
print("Accuracy:", accuracy_gam)

# Calculate precision, recall, and F1-score
precision_gam = precision_score(y_test, y_pred_gam)
recall_gam = recall_score(y_test, y_pred_gam)
f1_gam = f1_score(y_test, y_pred_gam)
print("Precision:", precision_gam)
print("Recall:", recall_gam)
print("F1-score:", f1_gam)

# LightGBM Model
print("\nLightGBM Model:")
train_data = lgb.Dataset(X, label=y)

# Define parameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'verbosity': -1
}

# Train LightGBM model
num_round = 100
bst = lgb.train(params, train_data, num_round)

# Predict probabilities on the test set
y_pred_proba_lgb = bst.predict(X_test_transformed_df)

# Calculate AUC
auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)
print("AUC:", auc_lgb)

# Convert predicted probabilities to binary predictions
y_pred_lgb = [1 if p > 0.97 else 0 for p in y_pred_proba_lgb]

# Calculate accuracy
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("Accuracy:", accuracy_lgb)

# Calculate precision, recall, and F1-score
precision_lgb = precision_score(y_test, y_pred_lgb)
recall_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
print("Precision:", precision_lgb)
print("Recall:", recall_lgb)
print("F1-score:", f1_lgb)

print(input("jd:"))