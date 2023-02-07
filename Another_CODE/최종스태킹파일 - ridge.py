import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor, LinearRegression


train = pd.read_parquet('./train_Final.parquet')
test = pd.read_parquet('./test_Final.parquet')
X = train.drop(['target'], axis=1)
y = train['target']
submit = pd.read_csv('./sample_submission.csv')

gbr_param = {'learning_rate': 0.0734430920138534, 'n_estimators': 275, 'max_depth': 10, 'min_samples_leaf': 12, 'min_samples_split': 13 , 'random_state' : 42}
hgb_param = {'loss': 'absolute_error', 'learning_rate': 0.0249816047538945, 'max_iter': 2877, 'max_leaf_nodes': 189, 'max_depth': 28, 'min_samples_leaf': 104, 'l2_regularization': 0.05, 'random_state': 42}
xgb_param = {'n_estimators': 1822, 'max_depth': 16, 'min_child_weight': 1, 'gamma': 1, 'learning_rate': 0.01, 'colsample_bytree': 0.7163696349835168, 'lambda': 0.015279229046863244, 'alpha': 0.0021628922658529686, 'subsample': 1.0}
cat_param = {'iterations': 589, 'learning_rate': 0.33865268174378255, 'depth': 16, 'min_data_in_leaf': 30, 'reg_lambda': 61.46317610780882, 'subsample': 0.7550764410597048, 'random_strength': 48.76480960330878, 'od_wait': 62, 'leaf_estimation_iterations': 6, 'bagging_temperature': 1.2692205293143553, 'colsample_bylevel': 0.14863052892541012}
lgbm_param = {'max_depth': 8, 'learning_rate': 0.005061576888752304, 'n_estimators': 3928, 'min_child_samples': 62, 'subsample': 0.46147273880126555, 'lambda_l1': 2.5348407664333426e-07, 'lambda_l2': 3.3323645788192616e-08, 'num_leaves': 222, 'feature_fraction': 0.7606690070459252, 'bagging_fraction': 0.8248435466776274, 'bagging_freq': 1}

# 메타 모델을 위한 학습 및 테스트 데이터 만들기
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=11)
    # 빈 배열 생성
    train_fold_pred = np.zeros((X_train_n.shape[0],1))
    test_pred = np.zeros((X_test_n.shape[0],n_folds))
    
    
    for folder_counter, (train_index, valid_index) in enumerate(skf.split(X_train_n,y)):
        print('폴드 세트 : ', folder_counter, ' 시작')
        print(len(X_train_n))
        print(train_index, valid_index)
        X_tr = X_train_n.loc[train_index]
        y_tr = y_train_n.loc[train_index]
        X_te = X_train_n.loc[valid_index] 
        
        # 폴드 내 모델 학습
        model.fit(X_tr, y_tr)
        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1) # y_train 예측, 폴드 끝나면 concat해야함
        test_pred[:, folder_counter] = model.predict(X_test_n) # y_test 예측, 폴드 끝나면 평균 낼거임

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)
    ####to_csv###
    train_fold = pd.DataFrame()
    test_pred = pd.DataFrame()
    
    train_fold = train_fold_pred
    test_pred = test_pred_mean
    
    train_fold = pd.DataFrame(train_fold)
    test_pred = pd.DataFrame(test_pred)


    #csv이름은 모델 이름마다 바꿔주기
    train_fold.to_csv('xgb_train_pred.csv', index = False)
    test_pred.to_csv('xgb_test_pred.csv', index = False)
    print("==================================To_csv complete============================================")
    print("==================================To_csv complete============================================")
    print("==================================To_csv complete============================================")
    print("==================================To_csv complete============================================")
    print("==================================To_csv complete============================================")
    #############
        
    return train_fold_pred, test_pred_mean # 하나의 모델에 대한 학습데이터, 테스트 데이터 생성


XGB = XGBRegressor(**xgb_param, tree_method='gpu_hist', gpu_id=0, random_state = 42)
lgbm = LGBMRegressor(**lgbm_param, random_state = 42)
Cat = CatBoostRegressor(**cat_param, random_state = 9555,eval_metric="MAE",)
hgr = HistGradientBoostingRegressor(**hgb_param)

# 개별 모델로부터 메타 모델에 필요한 데이터 셋 만들기
xgb_train_pred, xgb_test_pred = get_stacking_base_datasets(XGB, X, y, test, 7)
Cat_train_pred, Cat_test_pred = get_stacking_base_datasets(Cat, X, y, test, 7)
lgbm_train_pred, lgbm_test_pred = get_stacking_base_datasets(lgbm, X, y, test, 7)
hgr_train_pred, hgr_test_pred = get_stacking_base_datasets(hgr, X, y, test, 7)



xgb_train_pred = pd.read_csv('xgb_train_pred.csv')
xgb_train_pred = xgb_train_pred.to_numpy()

Cat_train_pred = pd.read_csv('cat_train_pred.csv')
Cat_train_pred = Cat_train_pred.to_numpy()

lgbm_train_pred = pd.read_csv('lgbm_train_pred.csv')
lgbm_train_pred = lgbm_train_pred.to_numpy()

hgr_train_pred = pd.read_csv('hgr_train_pred.csv')
hgr_train_pred = hgr_train_pred.to_numpy()


xgb_test_pred = pd.read_csv('xgb_test_pred.csv')
xgb_test_pred = xgb_test_pred.to_numpy()

Cat_test_pred = pd.read_csv('cat_test_pred.csv')
Cat_test_pred = Cat_test_pred.to_numpy()

lgbm_test_pred = pd.read_csv('lgbm_test_pred.csv')
lgbm_test_pred = lgbm_test_pred.to_numpy()

hgr_test_pred = pd.read_csv('hgr_test_pred.csv')
hgr_test_pred = hgr_test_pred.to_numpy()

print(xgb_train_pred)
print(xgb_test_pred)


# 개별 모델로부터 나온 y_train 예측값들 옆으로 붙이기
Stack_final_X_train = np.concatenate((xgb_train_pred,Cat_train_pred,lgbm_train_pred,hgr_train_pred), axis=1)
# 개별 모델로부터 나온 y_test 예측값들 옆으로 붙이기
Stack_final_X_test = np.concatenate((xgb_test_pred,Cat_test_pred,lgbm_test_pred,hgr_test_pred), axis=1)


lr_final = Ridge(alpha = 0.9)
lr_final.fit(Stack_final_X_train, y)
stack_final = lr_final.predict(Stack_final_X_test)

submit['target'] = stack_final
submit.to_csv('stacking_meta_ridge9555.csv', index = False)
