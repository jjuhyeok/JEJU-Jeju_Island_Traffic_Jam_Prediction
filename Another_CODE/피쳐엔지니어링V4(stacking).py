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
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

train_x = pd.read_csv('./train_x_309plus3_2.csv')
train_y = pd.read_csv('./train_y_309plus3_2.csv')
test_x = pd.read_csv('./test_x_309plus3_2.csv')


seed = 1422
# estimator1 = LinearRegression(n_jobs=-1)
# estimator2 = Ridge(random_state=seed)
# estimator3 = Lasso(random_state=seed)
estimator1 = ElasticNet(normalize =True, random_state=seed) #13.1205
# estimator5 = LassoLars(random_state=seed)
# estimator6 = BayesianRidge()
# estimator7 = KernelRidge()
# estimator2 = KNeighborsRegressor(n_neighbors = 10, n_jobs=-1)
# estimator3= DecisionTreeRegressor(random_state=seed)
# estimator3= ExtraTreeRegressor(random_state=seed)
# estimator4= BaggingRegressor(n_jobs=-1, random_state=seed)
#estimator2= ExtraTreesRegressor(n_jobs=-1, random_state=seed)
#estimator2= RandomForestRegressor(n_jobs=-1, random_state=seed)
#estimator3= GradientBoostingRegressor(random_state=seed)
estimator2= XGBRegressor(n_jobs=-1, random_state=seed) #3.589842529520123
estimator3= LGBMRegressor(n_jobs=-1, random_state=seed) #4.060863790742054
estimator4= CatBoostRegressor(verbose=False, random_state=seed) #3.377369174781055
estimator5= MLPRegressor(random_state=seed)



def mae_def(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))



def get_stacking_ml_datasets(model, train_x, train_y_n, n_folds):
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cnt = 0
    mae = 9999
    best_model = model
    scores = []
    cntt = 0
    best_score =0
    for folder_counter, (train_index, valid_index) in enumerate(kf.split(train_x, train_y_n)):
        cnt+=1
        print("===============%d fold==================="%(cnt))
        X_tr = train_x.iloc[train_index]
        y_tr = train_y_n.iloc[train_index]
        X_val = train_x.iloc[valid_index]
        y_val = train_y_n.iloc[valid_index]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        t_mae = mae_def(y_val, y_pred)
        scores.append(t_mae)
        if t_mae<mae:
            mae = t_mae
            best_score= t_mae
            best_model = model
    cntt+=1
    print("===============%d model complete==================="%(cntt))
    print(best_score)
    
    return best_model,best_score, np.mean(scores)
  




from tqdm import tqdm
base_ml = [
           estimator1, estimator2, estimator3, estimator4, estimator5
           ]

final_model = []
for i in tqdm(range(1)):
#     X_train, X_test, y_train, y_test = train_test_split(train_x, train_y['Y_'+str(i)], test_size = 0.2, random_state=1422)
    best_models = []
    scores={}
    
    for idx, estimator in tqdm(enumerate(base_ml)):
        temp_best_model, temp_best_score, temp_mean_score = get_stacking_ml_datasets(estimator, train_x, train_y['target'], 5)
        scores[idx] = temp_mean_score
        best_models.append(temp_best_model)
    
    sorted(scores.items(), key = lambda item : item[1])

    model_idx=np.array(sorted(scores.items(), key = lambda item : item[1]))[:3, 0]
    final_model.append([value for i, value in enumerate(best_models) if i in model_idx])
    print(final_model[int(i)-1])
