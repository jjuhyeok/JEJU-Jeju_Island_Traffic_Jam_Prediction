import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import lightgbm as lgb
from os.path import join
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna import Trial
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

X = pd.read_csv('train_x_309plus3_2.csv')
y = pd.read_csv('train_y_309plus3_2.csv')
test = pd.read_csv('test_x_309plus3_2.csv')

def objective(trial):
    '''
    param = {
        'objective': 'regression', # 회귀
        'metric': 'mae', 
        'max_depth': trial.suggest_int('max_depth',3, 16),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        'random_state': 42
    }
    '''
    
    param = {
        'objective': 'regression', # 회귀
        'verbose': -1,
        'metric': 'rmse', 
        'max_depth': trial.suggest_int('max_depth',3, 15),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
    }
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3, random_state=42)
    model = lgb.LGBMRegressor(**param)
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid,y_valid)],
            early_stopping_rounds=50,
            verbose=100)
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)

    return mae

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name = 'lgbm',
    direction = 'minimize',
    sampler = sampler
)

study.optimize(objective, n_trials=50)
print("Best Score:",study.best_value)
print("Best trial",study.best_trial.params)

params = study.best_trial.params
cat_model = lgb.LGBMRegressor(**params).fit(X, y)
y_pred = cat_model.predict(test)

import pandas as pd

sample_submission = pd.read_csv('./jeju_data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("./submit_lgbm_optuna.csv", index=False)




'''[32m[I 2022-11-06 17:32:14,747][0m Trial 13 finished with value: 3.184733213272421 and parameters: {'max_depth': 12, 'learning_rate': 0.009612192734398511, 'n_estimators': 3525, 'min_child_samples': 61, 'subsample': 0.5289687593895676, 'lambda_l1': 7.22020409650925, 'lambda_l2': 0.03316263661666914, 'num_leaves': 169, 'feature_fraction': 0.4704810679954663, 'bagging_fraction': 0.7352791498259887, 'bagging_freq': 3}. Best is trial 13 with value: 3.184733213272421.[0m
'''
