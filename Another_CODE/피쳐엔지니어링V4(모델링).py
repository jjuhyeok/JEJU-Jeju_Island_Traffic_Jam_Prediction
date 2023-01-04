import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc

from IPython.display import display


train_x = pd.read_csv('./EngineeringV4_train_x_2.csv')
train_y = pd.read_csv('./EngineeringV4_train_y_2.csv')
test_x = pd.read_csv('./EngineeringV4_test_x_2.csv')


print("====================Start Modeling======================")


#######################Modeling########################



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
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_jobs = -1)
rf_model.fit(train_x, train_y)
y_predict = rf_model.predict(test_x)
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission.to_csv("./submitV6_rf.csv", index = False)

print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")


#save_model
#save_model(final_model, 'my_best_pipeline_tuned_FE_V4', verbose=True)

#loaded_model = load_model('my_best_pipeline')
