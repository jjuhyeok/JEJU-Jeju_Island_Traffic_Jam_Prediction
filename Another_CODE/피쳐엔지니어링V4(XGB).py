from os.path import join
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

X = pd.read_csv('./train_x_V4.csv')
y = pd.read_csv('./train_y_V4.csv')
test = pd.read_csv('./test_x_V4.csv')

#다포함시키면 9번 돌았을때 2.98
#'location_cluster_41', '평균속도_mean', '평균속도_median','평균속도_mean_log', '평균속도_median_log', 
X.drop(['start_node_name', 'road_name', 'road_rating','road_in_use', '교통량 mean_log', '교통량 median_log','평균 소요 시간 mean', '평균 소요 시간 median', '평균 소요 시간 mean_log', '평균 소요 시간 median_log', '교통량 mean', '교통량 median','거주인구', '거주인구 median', '근무인구 mean', '근무인구 median', '방문인구 mean', '방문인구 median', '총 유동인구 mean', '총 유동인구 median', '거주인구_log', '거주인구 median_log', '근무인구 mean_log', '근무인구 median_log', '방문인구 mean_log', '방문인구 median_log', '총 유동인구 mean_log', '총 유동인구 median_log'], axis = 1 ,inplace = True)

test.drop(['start_node_name','road_name', 'road_rating', 'road_in_use', '교통량 mean_log', '교통량 median_log','평균 소요 시간 mean', '평균 소요 시간 median', '평균 소요 시간 mean_log', '평균 소요 시간 median_log', '교통량 mean', '교통량 median','거주인구', '거주인구 median', '근무인구 mean', '근무인구 median', '방문인구 mean', '방문인구 median', '총 유동인구 mean', '총 유동인구 median', '거주인구_log', '거주인구 median_log', '근무인구 mean_log', '근무인구 median_log', '방문인구 mean_log', '방문인구 median_log', '총 유동인구 mean_log', '총 유동인구 median_log'], axis = 1 ,inplace = True)

X['node_TF'][X['node_TF'] == True] = '1'
X['node_TF'][X['node_TF'] == False] = '0'
test['node_TF'][test['node_TF'] == True] = '1'
test['node_TF'][test['node_TF'] == False] = '0'

print(X['node_TF'])

X['node_TF'] = pd.to_numeric(X['node_TF'])
test['node_TF'] = pd.to_numeric(test['node_TF'])


#######팀원들과 이 부분 토의하기
#만약 이렇게 한다면 꼭 예측하는 부분에 103인덱스는 16추가해주기
#그럼 타겟값 빼는거는 안해줘야 맞지않나?
X = pd.concat([X,y], axis = 1)
X[X['connect_code'] == 103]['target'] - 16
y['target'] = X['target']
X = X.drop(['target'], axis = 1)
X.drop(['connect_code'], axis = 1, inplace = True)
test.drop(['connect_code'], axis = 1, inplace = True)



print(X.info())
print(y.info())
print(test.info())



def objective_xgb(trial: Trial, x, y):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 500, 5000),
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_int('gamma', 1, 3),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
        'random_state': 42
    }

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

    model = XGBRegressor(**params, tree_method='gpu_hist', gpu_id=0)
    xgb_model = model.fit(x_train, y_train, verbose = False, eval_set=[(x_val, y_val)], early_stopping_rounds=50)
    y_pred = xgb_model.predict(x_val)
    score = mean_absolute_error(y_val, y_pred)

    return score


study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=30)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))



#param = {'n_estimators': 4872, 'max_depth': 16, 'min_child_weight': 44, 'gamma': 2, 'learning_rate': 0.014, 'colsample_bytree': 0.5671944216803397, 'lambda': 0.001094611698221735, 'alpha': 0.40520461510161515, 'subsample': 1.0}


#2.962       pb : 3.09 / csv : train_x_309plus3_2
#param = {'n_estimators': 3076, 'max_depth': 15, 'min_child_weight': 4, 'gamma': 2, 'learning_rate': 0.008, 'colsample_bytree': 0.9368451773754232, 'lambda': 0.0673675007516321, 'alpha': 9.133401831074705, 'subsample': 0.7}

#2.95
#param = {'n_estimators': 3374, 'max_depth': 16, 'min_child_weight': 10, 'gamma': 3, 'learning_rate': 0.016, 'colsample_bytree': 0.8112588277346727, 'lambda': 3.7401404996137217, 'alpha': 0.0029622116733393662, 'subsample': 0.7}
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


param = study.best_trial.params

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []
y = y.astype(int)

for train_idx, val_idx in skf.split(X, y):
    folds.append((train_idx, val_idx))

XGB_model= {}

for f in range(5):
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

for fold in range(5):
    sample_submission['target'] += XGB_model[fold].predict(test)/5    
    
sample_submission.to_csv("./submit_xgb_fold_V2_259.csv", index=False)

sample_submission1 = pd.read_csv('./sample_submission.csv')
sample_submission2 = pd.read_csv('./sample_submission.csv')
sample_submission3 = pd.read_csv('./sample_submission.csv')
sample_submission4 = pd.read_csv('./sample_submission.csv')
sample_submission5 = pd.read_csv('./sample_submission.csv')

sample_submission1['target'] = XGB_model[0].predict(test) 
sample_submission2['target'] = XGB_model[1].predict(test)
sample_submission3['target'] = XGB_model[2].predict(test)
sample_submission4['target'] = XGB_model[3].predict(test)
sample_submission5['target'] = XGB_model[4].predict(test)

sample_submission1.to_csv("./submit_xgb_fold1_V2_259.csv", index=False)

sample_submission2.to_csv("./submit_xgb_fold2_V2_259.csv", index=False)

sample_submission3.to_csv("./submit_xgb_fold3_V2_259.csv", index=False)

sample_submission4.to_csv("./submit_xgb_fold4_V2_259.csv", index=False)

sample_submission5.to_csv("./submit_xgb_fold5_V2_259.csv", index=False)

df_imp = pd.DataFrame({'imp':XGB.feature_importances_}, index = XGB.feature_names_in_)
df_imp = df_imp[df_imp.imp > 0].sort_values('imp').copy()
print(df_imp)
