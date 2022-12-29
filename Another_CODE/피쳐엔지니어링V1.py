import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.model_selection import StratifiedKFold
import math
from IPython.display import display
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

train_x = pd.read_csv('./train_x_Team_merge2(speed40drop).csv')
train_y = pd.read_csv('./train_y_Team_merge2.csv')
test_x = pd.read_csv('./test_x_Team_merge2.csv')


#2
train_x.drop(['maximum_speed_limit_Eb','seasonb','road_rating','month','month_daya'],axis=1, inplace = True)
test_x.drop(['maximum_speed_limit_Eb','seasonb','road_rating','month','month_daya'],axis=1, inplace = True)

#3
train_x.drop(['start_node_name','end_node_name'],axis=1, inplace = True)
test_x.drop(['start_node_name','end_node_name'],axis=1, inplace = True)

#4
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_x['jeju_dist_MinMax'] = scaler.fit_transform(train_x['jeju_dist'].values.reshape(-1,1))
test_x['jeju_dist_MinMax'] = scaler.transform(test_x['jeju_dist'].values.reshape(-1,1))
scaler = MinMaxScaler()
train_x['seogwi_dist_MinMax'] = scaler.fit_transform(train_x['seogwi_dist'].values.reshape(-1,1))
test_x['seogwi_dist_MinMax'] = scaler.transform(test_x['seogwi_dist'].values.reshape(-1,1))
basic_col = ['seogwi_dist','jeju_dist']
for col in basic_col:
    print(col)
    train_x[col+"_log"] = train_x[col].apply(lambda x : math.log(x))

basic_col_t = ['seogwi_dist','jeju_dist']
for col in basic_col_t:
    test_x[col+"_log"] = test_x[col].apply(lambda x : math.log(x))


#5 이산형 변수들 로그취한거 드랍
train_x.drop(['maximum_speed_limit_log','weight_restricted_log', 'sm_tm_log', 'lane_ms_log','lane_rating_log','rating_ms_log','lazy_ms_log','weight_ms_log'],axis=1, inplace = True)
test_x.drop(['maximum_speed_limit_log','weight_restricted_log', 'sm_tm_log', 'lane_ms_log','lane_rating_log','rating_ms_log','lazy_ms_log','weight_ms_log'],axis=1, inplace = True)

#6 another 드랍
#train_x.drop(['lane_ms','lane_rating', 'rating_ms', 'lazy_ms','weight_ms'],axis=1, inplace = True)
#test_x.drop(['lane_ms','lane_rating', 'rating_ms', 'lazy_ms','weight_ms'],axis=1, inplace = True)

train_x.info()
test_x.info()

#train_x.drop(['start_turn_restricted', 'end_turn_restricted', 'turn_restricted' , 'start_cartesian' , 'end_cartesian'] , axis = 1, inplace = True)
#train_x.drop(['level_0', 'index', 'base_date'] , axis = 1, inplace = True)
#train_x.drop(['road_rating', 'month','month_day'] , axis = 1, inplace = True)


#test_x.drop(['start_turn_restricted', 'end_turn_restricted', 'turn_restricted' , 'start_cartesian' , 'end_cartesian'] , axis = 1, inplace = True)
#test_x.drop(['road_rating', 'month','month_day'] , axis = 1, inplace = True)
print("====================Start Modeling======================")


#######################Modeling########################

# Optuna를 통한 하이퍼 파라미터 설정
'''
def objective(trial):
    params = {
            'iterations':trial.suggest_int("iterations", 300, 1000),
            'learning_rate' : trial.suggest_uniform('learning_rate',0.1, 1),
            'depth': trial.suggest_int('depth',5, 16),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'reg_lambda': trial.suggest_uniform('reg_lambda',30,100),
            'subsample': trial.suggest_uniform('subsample',0.3,1),
            'random_strength': trial.suggest_uniform('random_strength',10,100),
            'od_wait':trial.suggest_int('od_wait', 10, 150),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,20),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 1, 100),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0., 1.0),
            'random_state' : 777,
            'verbose' : 0,
        }
    #'task_type' : 'GPU',
    #"eval_metric":'RMSE',
    x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=1234)
    cat = CatBoostRegressor(**params)
    cat.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_valid,y_valid)],
              verbose=False)
    cat_pred = cat.predict(x_valid)
    score = mean_absolute_error(y_valid, cat_pred)
    return score

study = optuna.create_study(study_name = 'cat_parameter_opt', direction = 'minimize', sampler = TPESampler(seed=1234))
study.optimize(objective, n_trials=70)
print("Best Score:", study.best_value)
print("Best trial", study.best_trial.params)
## 하이퍼 파리미터 저장
joblib.dump(study, "./cat_parameter_opt.pkl")
# Optuna로 탐색한 최적의 하이퍼 파라미터 load
study = joblib.load("./cat_parameter_opt.pkl") # 로드
study.best_params

'''


print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")
print("===========================study Finish=========================")





bootstrap_type='Poisson'

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


###############################################################
#Optuna용 Train셋

'''

train_y = train_y.astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for (train_idx, valid_idx) in (skf.split(train_x, train_y)):
  folds.append((train_idx,valid_idx))

cat_models={}

for fold in range(5):
  print(f'===================================={fold+1}============================================')
  train_idx, valid_idx = folds[fold]
  X_train = train_x.iloc[train_idx].values 
  X_valid = train_x.iloc[valid_idx].values
  y_train = train_y['target'][train_idx].values
  y_valid = train_y['target'][valid_idx].values

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
  cat.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid,y_valid)],
          early_stopping_rounds=35)
  cat_models[fold] = cat
  print(f'================================================================================\n\n')


submit = pd.read_csv('./sample_submission.csv')
for fold in range(5):
  submit['target'] += cat_models[fold].predict(test_x)
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

