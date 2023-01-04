import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.model_selection import StratifiedKFold
import math
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
# 모델
# 모델
import sklearn 
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import catboost
from catboost import CatBoostRegressor
import joblib
import warnings
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

train_x = pd.read_csv('./train_x_V4.csv')
train_y = pd.read_csv('./train_y_V4.csv')
test_x = pd.read_csv('./test_x_V4.csv')

#다포함시키면 9번 돌았을때 2.98
#'location_cluster_41', '평균속도_mean', '평균속도_median','평균속도_mean_log', '평균속도_median_log', 
train_x.drop([ '교통량 mean_log', '교통량 median_log','평균 소요 시간 mean', '평균 소요 시간 median', '평균 소요 시간 mean_log', '평균 소요 시간 median_log', '교통량 mean', '교통량 median','거주인구', '거주인구 median', '근무인구 mean', '근무인구 median', '방문인구 mean', '방문인구 median', '총 유동인구 mean', '총 유동인구 median', '거주인구_log', '거주인구 median_log', '근무인구 mean_log', '근무인구 median_log', '방문인구 mean_log', '방문인구 median_log', '총 유동인구 mean_log', '총 유동인구 median_log'], axis = 1 ,inplace = True)

test_x.drop([ '교통량 mean_log', '교통량 median_log','평균 소요 시간 mean', '평균 소요 시간 median', '평균 소요 시간 mean_log', '평균 소요 시간 median_log', '교통량 mean', '교통량 median','거주인구', '거주인구 median', '근무인구 mean', '근무인구 median', '방문인구 mean', '방문인구 median', '총 유동인구 mean', '총 유동인구 median', '거주인구_log', '거주인구 median_log', '근무인구 mean_log', '근무인구 median_log', '방문인구 mean_log', '방문인구 median_log', '총 유동인구 mean_log', '총 유동인구 median_log'], axis = 1 ,inplace = True)

train_x['node_TF'][train_x['node_TF'] == True] = '1'
train_x['node_TF'][train_x['node_TF'] == False] = '0'
test_x['node_TF'][test_x['node_TF'] == True] = '1'
test_x['node_TF'][test_x['node_TF'] == False] = '0'

print(train_x['node_TF'])

train_x['node_TF'] = pd.to_numeric(train_x['node_TF'])
test_x['node_TF'] = pd.to_numeric(test_x['node_TF'])


print("====================Start Modeling======================")


#######################Modeling########################

# Optuna를 통한 하이퍼 파라미터 설정

def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 1, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
        }
    #'task_type' : 'GPU',
    #"eval_metric":'RMSE',
    x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=1234)
    rf = RandomForestRegressor(**params, random_state=1234)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_valid)
    score = mean_absolute_error(y_valid, rf_pred)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
print("Best Score:", study.best_value)
print("Best trial", study.best_trial.params)
## 하이퍼 파리미터 저장
joblib.dump(study, "./rf_parameter_opt.pkl")
# Optuna로 탐색한 최적의 하이퍼 파라미터 load
study = joblib.load("./rf_parameter_opt.pkl") # 로드
study.best_params




print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")



'''

bootstrap_type='Poisson'

#merge2CSV hyperparameter
cat = CatBoostRegressor(iterations=541,
                    learning_rate= 0.20783283110330608,
                    depth=16,
                    min_data_in_leaf=10,
                    reg_lambda=81.86892649577524,
                    subsample=0.6512518642490095,    #gpu 지원 x
                    random_strength=47.65169280068649,
                    od_wait=102,
                    leaf_estimation_iterations=11,
                    bagging_temperature=49.03900542494145,
                    colsample_bylevel=0.4829046416508313, #gpu 지원 x
                    eval_metric="MAE",
                    #task_type = "GPU",
                    )

                    '''

'''
#merge1CSV hyperparameter
cat = CatBoostRegressor(iterations=845,
                    learning_rate= 0.3263241956839758,
                    depth=14,
                    min_data_in_leaf=10,
                    reg_lambda=56.3058889494273,
                    subsample=0.7170389290299723,    #gpu 지원 x
                    random_strength=79.69489466120987,
                    od_wait=138,
                    leaf_estimation_iterations=9,
                    bagging_temperature=10.047254477842163,
                    colsample_bylevel=0.235269659582944, #gpu 지원 x
                    eval_metric="MAE",
                    #task_type = "GPU",
                    )
'''

###############################################################
#Optuna용 Train셋


train_y = train_y.astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for (train_idx, valid_idx) in (skf.split(train_x, train_y)):
  folds.append((train_idx,valid_idx))

rf_models={}

for fold in range(5):
  print(f'===================================={fold+1}============================================')
  train_idx, valid_idx = folds[fold]
  X_train = train_x.iloc[train_idx].values 
  X_valid = train_x.iloc[valid_idx].values
  y_train = train_y['target'][train_idx].values
  y_valid = train_y['target'][valid_idx].values

  #merge2CSV hyperparameter
  rf = RandomForestRegressor(iterations=541,
                        learning_rate= 0.20783283110330608,
                        depth=16,
                        min_data_in_leaf=10,
                        reg_lambda=81.86892649577524,
                        subsample=0.6512518642490095,    #gpu 지원 x
                        random_strength=47.65169280068649,
                        od_wait=102,
                        leaf_estimation_iterations=11,
                        bagging_temperature=49.03900542494145,
                        colsample_bylevel=0.4829046416508313, #gpu 지원 x
                        eval_metric="MAE",
                        #task_type = "GPU",
                        )

  rf.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid,y_valid)],
          early_stopping_rounds=35)
  rf_models[fold] = rf
  print(f'================================================================================\n\n')


submit = pd.read_csv('./sample_submission.csv')
for fold in range(5):
  submit['target'] += rf_models[fold].predict(test_x)
submit['target'] = submit['target']/5
submit.to_csv("./submit_AFE3.csv", index = False)

'''


cat.fit(train_x,train_y)


prediction=cat.predict(test_x)
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = prediction
sample_submission.to_csv("./submit_AFE3_drop40_nofold.csv", index = False)

print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")



#save_model
#save_model(final_model, 'my_best_pipeline_tuned_FE_V4', verbose=True)

#loaded_model = load_model('my_best_pipeline')

'''
