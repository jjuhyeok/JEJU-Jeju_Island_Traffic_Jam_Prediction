import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc

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

train_x = pd.read_csv('./EngineeringV4_train_x_2.csv')
train_y = pd.read_csv('./EngineeringV4_train_y_2.csv')
test_x = pd.read_csv('./EngineeringV4_test_x_2.csv')


print("====================Start Modeling======================")


#######################Modeling########################
'''

# Optuna를 통한 하이퍼 파라미터 설정

def objective(trial):
    params = {
            'iterations':trial.suggest_int("iterations", 100, 2000),
            'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
            'depth': trial.suggest_int('depth',11, 16),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'reg_lambda': trial.suggest_uniform('reg_lambda',30,100),
            #'subsample': trial.suggest_uniform('subsample',0.01,10),
            'random_strength': trial.suggest_uniform('random_strength',10,100),
            'od_wait':trial.suggest_int('od_wait', 10, 150),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,20),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 1, 100),
            #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0., 1.0),
            'random_state' : 1234,
            'verbose' : 0,
            'task_type' : 'GPU',
            "eval_metric":'RMSE',
        }

    x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=1234)
    cat = CatBoostRegressor(**params)
    cat.fit(x_train, y_train, eval_set=[(x_valid,y_valid)],
              verbose=False)
    cat_pred = cat.predict(x_valid)
    score = mean_absolute_error(y_valid, cat_pred)
    return score

study = optuna.create_study(study_name = 'cat_parameter_opt', direction = 'minimize', sampler = TPESampler(seed=1234))
study.optimize(objective, n_trials=150)
print("Best Score:", study.best_value)
print("Best trial", study.best_trial.params)
## 하이퍼 파리미터 저장
joblib.dump(study, "./cat_parameter_opt.pkl")
# Optuna로 탐색한 최적의 하이퍼 파라미터 load
study = joblib.load("./cat_parameter_opt.pkl") # 로드
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

'''
bootstrap_type='Poisson'


cat = CatBoostRegressor(iterations=1886,
                        learning_rate= 0.17802732849999442,
                        depth=12,
                        min_data_in_leaf=25,
                        reg_lambda=97.49081346385762,
                        subsample=0.75,    #gpu 지원 x
                        random_strength=25.8430651731408,
                        od_wait=56,
                        leaf_estimation_iterations=8,
                        bagging_temperature=1.1472603272063462,
                        colsample_bylevel=0.83, #gpu 지원 x
                        eval_metric="MAE",
                        #task_type = "GPU",
                        )
'''
'''
cat = CatBoostRegressor(iterations=1000,
                        learning_rate= 0.33,
                        depth=16,
                        min_data_in_leaf=7,
                        reg_lambda=38,
                        #subsample=0.75,    #gpu 지원 x
                        random_strength=48,
                        od_wait=107,
                        leaf_estimation_iterations=10,
                        bagging_temperature=10.6,
                        #colsample_bylevel=0.83, #gpu 지원 x
                        #n_jobs = -1
                        eval_metric="MAE",
                        task_type = "GPU",
                        )'''


'''
cat.fit(train_x,train_y)
'''

'''from pycaret.regression import *

caret = pd.concat([train_x,train_y], axis = 1)



model = setup(caret, target = 'target',  session_id = 42, silent = True, use_gpu = True, fold = 5, html=False)



#rf는 여기서 안되는거같은데
#best_3 = compare_models(include = [ 'dt', 'lightgbm' , 'catboost', 'xgboost','rf'], n_select = 3, fold = 5, sort = 'MAE')
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")

total_models = []
#gbr = create_model('gbr', fold = 5)
#et = create_model('et', fold = 5)
#ada = create_model('ada', fold = 5)
#rf = create_model('rf', fold = 5)

model_dt = create_model('dt', fold = 5)
model_xgboost = create_model('xgboost', fold = 5)
model_catboost = create_model('catboost', fold = 5)
print("==================create_model_complete========================")
print("==================create_model_complete========================")
print("==================create_model_complete========================")
print("==================create_model_complete========================")
print("==================create_model_complete========================")
print("==================create_model_complete========================")



#model_dt_tune = tune_model(model_dt, fold=5, optimize = 'MAE', choose_better = True, verbose = True)
print("==================tuned_model_1_complete========================")
print("==================tuned_model_1_complete========================")
print("==================tuned_model_1_complete========================")
print("==================tuned_model_1_complete========================")
print("==================tuned_model_1_complete========================")
model_xgboost_tune = tune_model(model_xgboost, fold=5, optimize = 'MAE', choose_better = True, verbose = True)
print("==================tuned_model_2_complete========================")
print("==================tuned_model_2_complete========================")
print("==================tuned_model_2_complete========================")
print("==================tuned_model_2_complete========================")
print("==================tuned_model_2_complete========================")
#model_catboost_tune = tune_model(model_catboost, fold=5, optimize = 'MAE', choose_better = True, verbose = True)
print("==================tuned_model_3_complete========================")
print("==================tuned_model_3_complete========================")
print("==================tuned_model_3_complete========================")
print("==================tuned_model_3_complete========================")
print("==================tuned_model_3_complete========================")
print("==================tuned_model_3_complete========================")

total_models.append(model_dt)
total_models.append(model_xgboost_tune)
total_models.append(model_catboost)



#tuned_best_3 = [tune_model(i, optimize = 'MAE', fold = 5,  verbose = True) for i in best_3]
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
#lightgbm = create_model('lightgbm')
#cat = create_model('catboost')
dt = create_model('dt', fold = 5)
stacker = stack_models(total_models,fold = 5, meta_model = dt)
print("==================stacking_complete========================")
print("==================stacking_complete========================")
print("==================stacking_complete========================")
print("==================stacking_complete========================")
print("==================stacking_complete========================")
final_model = finalize_model(stacker)
print("==================finalize_model_complete========================")
print("==================finalize_model_complete========================")
print("==================finalize_model_complete========================")
print("==================finalize_model_complete========================")
print("==================finalize_model_complete========================")


#final_model = load_model('my_best_pipeline_tuned(dt는튜닝x+xgb튜닝)+lgbm스태킹')

print("==================load_model_complete========================")
print("==================load_model_complete========================")
print("==================load_model_complete========================")
print("==================load_model_complete========================")
print("==================load_model_complete========================")

y_predict = predict_model(final_model, data=test_x)


#######################submission########################
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = y_predict['Label']
sample_submission.to_csv("./submitV5_stack_dt.csv", index = False)

'''


'''
prediction=cat.predict(test_x)
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = prediction
sample_submission.to_csv("./submitV6_catV2.csv", index = False)

print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
'''
from st_aggrid import AgGrid
import streamlit as st
import streamlit.components.v1 as components
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.0002)

model=CatBoostRegressor(silent=True, random_state=0).fit(x_train, y_train)

perm = PermutationImportance(model, random_state=3).fit(x_valid, y_valid)

from streamlit import components
html_object = eli5.show_weights(perm, feature_names = x_valid.columns.tolist(), top=100)

raw_html = html_object._repr_html_()

components.v1.html(raw_html)

import shap

evaluate_model(model)

plot_model(model, plot = 'feature')

interpret_model(model)

#save_model
#save_model(final_model, 'my_best_pipeline_tuned_FE_V4', verbose=True)

#loaded_model = load_model('my_best_pipeline')


