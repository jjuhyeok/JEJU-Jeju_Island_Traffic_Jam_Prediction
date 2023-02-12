from os.path import join
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

X = pd.read_parquet('./train_57_jh_V2_1_drop.parquet')
y = X['target']
test = pd.read_parquet('./test_57_jh_V2_1_drop.parquet')

X.drop(['target'], axis = 1, inplace = True)


X.drop(['tour_count'], axis = 1, inplace = True)
test.drop(['tour_count'], axis = 1, inplace = True)


'''

X['node_TF'][X['node_TF'] == True] = '1'
X['node_TF'][X['node_TF'] == False] = '0'
test['node_TF'][test['node_TF'] == True] = '1'
test['node_TF'][test['node_TF'] == False] = '0'

print(X['node_TF'])

X['node_TF'] = pd.to_numeric(X['node_TF'])
test['node_TF'] = pd.to_numeric(test['node_TF'])
'''

print(X.info())
print(y.info())
print(test.info())


'''
def objective_xgb(trial: Trial, x, y):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_int('gamma', 1, 3),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.01,0.05,0.077, 0.1,0.2]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.3, 0.5, 0.75, 1.0]),
        'random_state': 42
    }

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state = 42)

    model = XGBRegressor(**params, tree_method='gpu_hist', gpu_id=0)
    xgb_model = model.fit(x_train, y_train, verbose = False, eval_set=[(x_val, y_val)], early_stopping_rounds=50)
    y_pred = xgb_model.predict(x_val)
    score = mean_absolute_error(y_val, y_pred)

    return score


study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=30)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))


'''

##########3.10
param = {'n_estimators': 1522, 'max_depth': 16, 'min_child_weight': 22, 'gamma': 3, 'learning_rate': 0.077, 'colsample_bytree': 0.7940599445756948, 'lambda': 0.1170745154259616, 'alpha': 4.579149590635459, 'subsample': 1.0}



##########1108 V2_1
#param = {'n_estimators': 4978, 'max_depth': 14, 'min_child_weight': 28, 'gamma': 2, 'learning_rate': 0.03, 'colsample_bytree': 0.9291993066735358, 'lambda': 3.590954962627517, 'alpha': 0.15018434208490658, 'subsample': 0.75}       



##########1108 V2
#param = {'n_estimators': 2720, 'max_depth': 15, 'min_child_weight': 8, 'gamma': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.8624431739130307, 'lambda': 0.04597085750106583, 'alpha': 0.0038989006917591177, 'subsample': 0.75}



#param = {'n_estimators': 3298, 'max_depth': 16, 'min_child_weight': 29, 'gamma': 2, 'learning_rate': 0.008, 'colsample_bytree': 0.7769376556689326, 'lambda': 0.8363851666966314, 'alpha': 0.006385408919207994, 'subsample': 0.8}



#param = {'n_estimators': 4872, 'max_depth': 16, 'min_child_weight': 44, 'gamma': 2, 'learning_rate': 0.014, 'colsample_bytree': 0.5671944216803397, 'lambda': 0.001094611698221735, 'alpha': 0.40520461510161515, 'subsample': 1.0}


#2.962       pb : 3.09 / csv : train_x_309plus3_2
#param = {'n_estimators': 3076, 'max_depth': 15, 'min_child_weight': 4, 'gamma': 2, 'learning_rate': 0.008, 'colsample_bytree': 0.9368451773754232, 'lambda': 0.0673675007516321, 'alpha': 9.133401831074705, 'subsample': 0.7}

#2.95
#param = {'n_estimators': 3374, 'max_depth': 16, 'min_child_weight': 10, 'gamma': 3, 'learning_rate': 0.016, 'colsample_bytree': 0.8112588277346727, 'lambda': 3.7401404996137217, 'alpha': 0.0029622116733393662, 'subsample': 0.7}
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


#param = study.best_trial.params

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

folds = []
y = y.astype(int)

for train_idx, val_idx in skf.split(X, y):
    folds.append((train_idx, val_idx))

XGB_model= {}

for f in range(10):
      print(f'===================================={f+1}============================================')
      train_idx, val_idx = folds[f]
      
      x_train, x_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
      
      XGB = XGBRegressor(**param, tree_method='gpu_hist', gpu_id=0)
      XGB.fit(x_train, y_train)
      
      y_pred = XGB.predict(x_val)
      mae = mean_absolute_error(y_val, y_pred)
      print(f"{f + 1} Fold MAE = {mae}")
      XGB_model[f] = XGB
      print(f'================================================================================\n\n')
              


sample_submission = pd.read_csv('./sample_submission.csv')

for fold in range(10):
    sample_submission['target'] += XGB_model[fold].predict(test)/10    
    
sample_submission.to_csv("./submit_xgb_fold_FE2.csv", index=False)

sample_submission1 = pd.read_csv('./sample_submission.csv')
sample_submission2 = pd.read_csv('./sample_submission.csv')
sample_submission3 = pd.read_csv('./sample_submission.csv')
sample_submission4 = pd.read_csv('./sample_submission.csv')
sample_submission5 = pd.read_csv('./sample_submission.csv')
sample_submission6 = pd.read_csv('./sample_submission.csv')
sample_submission7 = pd.read_csv('./sample_submission.csv')
sample_submission8 = pd.read_csv('./sample_submission.csv')
sample_submission9 = pd.read_csv('./sample_submission.csv')
sample_submission10 = pd.read_csv('./sample_submission.csv')

sample_submission1['target'] = XGB_model[0].predict(test) 
sample_submission2['target'] = XGB_model[1].predict(test)
sample_submission3['target'] = XGB_model[2].predict(test)
sample_submission4['target'] = XGB_model[3].predict(test)
sample_submission5['target'] = XGB_model[4].predict(test)
sample_submission6['target'] = XGB_model[5].predict(test) 
sample_submission7['target'] = XGB_model[6].predict(test)
sample_submission8['target'] = XGB_model[7].predict(test)
sample_submission9['target'] = XGB_model[8].predict(test)
sample_submission10['target'] = XGB_model[9].predict(test)

sample_submission1.to_csv("./submit_xgb_fold1_FE2.csv", index=False)

sample_submission2.to_csv("./submit_xgb_fold2_FE2.csv", index=False)

sample_submission3.to_csv("./submit_xgb_fold3_FE2.csv", index=False)

sample_submission4.to_csv("./submit_xgb_fold4_FE2.csv", index=False)

sample_submission5.to_csv("./submit_xgb_fold5_FE2.csv", index=False)

sample_submission6.to_csv("./submit_xgb_fold6_FE2.csv", index=False)

sample_submission7.to_csv("./submit_xgb_fold7_FE2.csv", index=False)

sample_submission8.to_csv("./submit_xgb_fold8_FE2.csv", index=False)

sample_submission9.to_csv("./submit_xgb_fold9_FE2.csv", index=False)

sample_submission10.to_csv("./submit_xgb_fold10_FE2.csv", index=False)

df_imp = pd.DataFrame({'imp':XGB.feature_importances_}, index = XGB.feature_names_in_)
df_imp = df_imp[df_imp.imp > 0].sort_values('imp').copy()
print(df_imp)
