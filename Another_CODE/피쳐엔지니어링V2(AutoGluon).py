#AutoGluon 쓸때는 numpy 1.23으로
#pycaret 쓸때는 numpy 1.20으로

import os
import torch
import random
import numpy as np
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, train_test_split

class CFG:
    dataPath = './'
    # trainPath = dataPath+'raw/new_train.csv'
    # testPath = dataPath+'raw/new_test.csv'
    trainYPath = dataPath+'EngineeringV4_train_y_1.csv'
    trainXPath = dataPath+'EngineeringV4_train_x_1.csv'
    testXPath = dataPath+'EngineeringV4_test_x_1.csv'

    #trainYPath = pd.read_csv('C:/Users/USER/Desktop/LG_Aimers/code/data/train.csv')
    #trainXPath = pd.read_csv('C:/Users/USER/Desktop/LG_Aimers/code/data/Newtrain_x.csv')
    #testXPath = pd.read_csv('C:/Users/USER/Desktop//LG_Aimers/code/data/Newsubmit_x.csv')


    submission = dataPath+'sample_submission.csv'
    outPath = dataPath+'processed/'
    #submission = pd.read_csv('C:/Users/USER/Desktop/LG_Aimers/code/data/sample_submission.csv')
    #outPath = dataPath+'processed/'
    # drop_list = ['X_04', 'X_23', 'X_47', 'X_48', 'X_10', 'X_11', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']
    fold_num = 10
    seed = 42



train_x = pd.read_csv('./EngineeringV4_train_x_1.csv')
train_y = pd.read_csv('./EngineeringV4_train_y_1.csv')
test_x = pd.read_csv('./EngineeringV4_test_x_1.csv')

print("====================Start Modeling======================")

kf = KFold(n_splits=2, shuffle=True, random_state=42)

for f_idx, (train_idx, val_idx) in enumerate(kf.split(train_x)):
    train_input, train_target = train_x.iloc[train_idx, :].copy(), train_y.iloc[train_idx, :].copy()
    val_input, val_target = train_x.iloc[val_idx, :].copy(), train_y.iloc[val_idx, :].copy()
    submit_test = pd.read_csv(CFG.submission).copy()
    submit_val = pd.DataFrame()
    submit_val['target'] = pd.read_csv(CFG.trainYPath)['target'].copy()

    fold_save_path = f'./FOLD{f_idx+1}/'
    os.makedirs(fold_save_path, exist_ok=True)
    
    train_data = pd.concat([train_input, train_target.iloc[:,0]], axis=1)
    val_data = pd.concat([val_input, val_target.iloc[:,0]], axis=1)
    y_true = val_data['target']
    val_data = val_data.drop(columns=['target'])

    save_path = fold_save_path + 'target'+ 'Models-predict'
    # good qulity -> 1.93471 LB 16
    # medium qulity ->
    print("predict")
    print("predict")
    predictor = TabularPredictor(label='target',  eval_metric='mean_absolute_error').fit(train_data, presets='high_quality',  ag_args_fit={'num_gpus': 1})
    # you can access with good_quilty

    y_pred_val = predictor.predict(val_data)
    # perf = predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True)
    y_pred_val_df = pd.DataFrame(y_pred_val, columns=['target'])

    # 제출 csv 추출
    y_pred_test = predictor.predict(test_x)
    y_pred_test_df = pd.DataFrame(y_pred_test, columns=['target'])

    submit_val[col] = y_pred_val_df
    submit_test[col] = y_pred_test_df
    submit_test.to_csv(f'./fold{f_idx+1}_submit_test.csv', index=False)
    submit_test = pd.read_csv(f'./fold{f_idx+1}_submit_test.csv')



    

print(y_number + ' Done **************************************************************************************************************')

#######################Modeling########################


train = pd.concat([train_x,train_y], axis = 1)



train_data = TabularDataset(train)
test_data = TabularDataset(test_x)

#train_data = TabularDataset('./train_data.csv')
#test_data = TabularDataset('./test_data.csv')


print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")

#save_path = 'Models-predict'  # specifies folder to store trained models

predictor = TabularPredictor(label='target',  eval_metric='mean_absolute_error').fit(train_data, presets='high_quality',  ag_args_fit={'num_gpus': 1})


print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")


#predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data)

print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")

y_pred_final = pd.DataFrame(y_pred, columns=['target'])

#######################submission########################
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = y_pred_final
sample_submission.to_csv("./submit5_auto.csv", index = False)


print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")
print("==================submission_complete========================")


'''
predictor.leaderboard(test_data, silent=True)

print("==================leaderboard_complete========================")
print("==================leaderboard_complete========================")
print("==================leaderboard_complete========================")
print("==================leaderboard_complete========================")
print("==================leaderboard_complete========================")


#perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



results = predictor.fit_summary(show_plot=True)
print("==================fit_summary_complete========================")
print("==================fit_summary_complete========================")
print("==================fit_summary_complete========================")
print("==================fit_summary_complete========================")
print("==================fit_summary_complete========================")
print(results)

predictor.leaderboard(test_data, silent=True, html = False)


'''

