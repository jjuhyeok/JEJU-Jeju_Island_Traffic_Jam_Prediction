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

X = pd.read_csv('./train_x_309plus3_2.csv')
y = pd.read_csv('./train_y_309plus3_2.csv')
test = pd.read_csv('./test_x_309plus3_2.csv')

X.drop(['road_in_use'], axis = 1, inplace = True)
test.drop(['road_in_use'], axis = 1,inplace = True)

############################Test2#############################################
coord = X[['start_latitude','start_longitude']]
coord_t = test[['start_latitude','start_longitude']]

pca = PCA(n_components=2)
pca.fit(coord)

coord_pca = pca.fit_transform(coord)
coord_pca_t = pca.transform(coord_t)

X['coord_pca1_s'] = coord_pca[:, 0]
X['coord_pca2_s'] = coord_pca[:, 1]

test['coord_pca1_s'] = coord_pca_t[:, 0]
test['coord_pca2_s'] = coord_pca_t[:, 1]


pca = PCA(n_components=1)
pca.fit(coord)

ca = pca.fit_transform(coord)
cat = pca.transform(coord_t)

X['ca'] = ca[:, 0]


test['ca'] = cat[:, 0]



coord = X[['end_latitude','end_longitude']]
coord_t = test[['end_latitude','end_longitude']]

pca = PCA(n_components=2)
pca.fit(coord)

coord_pca = pca.fit_transform(coord)
coord_pca_t = pca.transform(coord_t)

X['coord_pca1_e'] = coord_pca[:, 0]
X['coord_pca2_e'] = coord_pca[:, 1]

test['coord_pca1_e'] = coord_pca_t[:, 0]
test['coord_pca2_e'] = coord_pca_t[:, 1]



pca = PCA(n_components=1)
pca.fit(coord)

ca = pca.fit_transform(coord)
cat = pca.transform(coord_t)

X['ca2'] = ca[:, 0]


test['ca2'] = cat[:, 0]



cluster_centers = np.array([[33.2799141, 126.7207056], [33.2482109, 126.5113278], [33.2268151, 126.2522254], [33.2507866, 126.477585], [33.2578824, 126.5689267], [33.2558577, 126.5602063], [33.3810625, 126.8767558], [37.691055, 126.753565], [33.2496801, 126.3371482], [33.2687729, 126.5867363], [33.2542814, 126.3979513], [33.2458588, 126.5652696], [33.2539386, 126.4339728], [37.4841833, 126.9497246], [33.2477973, 126.5613541], [33.326494, 126.831638], [33.3099343, 126.7674357], [33.5170777, 126.5445896], [33.5225711, 126.8520445], [33.4830731, 126.4772131], [33.5028456, 126.4683154], [33.4916823, 126.5947483], [33.503187, 126.519751], [33.5116788, 126.5222179], [33.521762, 126.5855263], [33.4762704, 126.5450898], [33.4619478, 126.3295244], [33.4881534, 126.4969519], [33.495125, 126.5116449], [33.4928462, 126.4322356], [33.5096339, 126.514017], [33.5114842, 126.5117556], [33.5067302, 126.5270745], [33.4969863, 126.5353375], [33.4997903, 126.4582053], [33.5150396, 126.526393], [33.5116235, 126.5383642], [33.5344076, 126.6342455], [33.3501173, 126.184217], [33.4103985, 126.2671535], [33.5202063, 126.5655282]])

def make_cluster_41(train, test):
    from sklearn.cluster import KMeans
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    k_mean = KMeans(n_clusters=41, init=cluster_centers, random_state=np.random.RandomState(seed=14))
    #k_mean.fit(train_c)
    train['location_41cluster'] = k_mean.fit_predict(train_c)
    test['location_41cluster'] = k_mean.predict(test_c)
    return train, test

X, test = make_cluster_41(X, test)


'''from sklearn.preprocessing import OneHotEncoder

X['location_41cluster'] = X['location_41cluster'].astype(str)
test['location_41cluster'] = test['location_41cluster'].astype(str)

ohe = OneHotEncoder(sparse=False)
train_cat = ohe.fit_transform(X[['location_41cluster']])
print(ohe.categories_[0])
pd.DataFrame(train_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])
X = pd.concat([X.drop(['location_41cluster'], axis = 1), pd.DataFrame(train_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])], axis=1)


# fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
test_cat = ohe.transform(test[['location_41cluster']])
pd.DataFrame(test_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])
test = pd.concat([test.drop(['location_41cluster'], axis = 1), pd.DataFrame(test_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])], axis=1)


'''


X['Slong'] = X['start_longitude'] - ((X['start_longitude'].max() + X['start_longitude'].min())/2)
test['Slong'] = test['start_longitude'] - ((test['start_longitude'].max() + test['start_longitude'].min())/2)

X['Slat'] = X['start_latitude'] - ((X['start_latitude'].max() + X['start_latitude'].min())/2)
test['Slat'] = test['start_latitude'] - ((test['start_latitude'].max() + test['start_latitude'].min())/2)


X['Elong'] = X['end_longitude'] - ((X['end_longitude'].max() + X['end_longitude'].min())/2)
test['Elong'] = test['end_longitude'] - ((test['end_longitude'].max() + test['end_longitude'].min())/2)

X['Elat'] = X['end_latitude'] - ((X['end_latitude'].max() + X['end_latitude'].min())/2)
test['Elat'] = test['end_latitude'] - ((test['end_latitude'].max() + test['end_latitude'].min())/2)



'''
X.drop(['start_node_name', 'end_node_name'], axis = 1, inplace = True)
test.drop(['start_node_name', 'end_node_name'], axis = 1, inplace = True)
'''
############################Test1#############################################
############################성능 안좋아짐#####################################
'''
print(len(X))
train = pd.concat([X,y], axis = 1)
train = train[train['connect_code'] == 0]
y = train['target']
X = train.drop(['target'], axis = 1)
print(len(X))

X.drop(['connect_code', 'road_in_use'], axis = 1, inplace = True)
test.drop(['connect_code', 'road_in_use'], axis = 1, inplace = True)
'''
##############################################################################

X['node_TF'][X['node_TF'] == True] = '1'
X['node_TF'][X['node_TF'] == False] = '0'
test['node_TF'][test['node_TF'] == True] = '1'
test['node_TF'][test['node_TF'] == False] = '0'

print(X['node_TF'])

X['node_TF'] = pd.to_numeric(X['node_TF'])
test['node_TF'] = pd.to_numeric(test['node_TF'])


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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state = 42)

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
    
sample_submission.to_csv("./submit_xgb_fold_V2_drop103.csv", index=False)

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

sample_submission1.to_csv("./submit_xgb_fold1_V_drop103.csv", index=False)

sample_submission2.to_csv("./submit_xgb_fold2_V_drop103.csv", index=False)

sample_submission3.to_csv("./submit_xgb_fold3_V_drop103.csv", index=False)

sample_submission4.to_csv("./submit_xgb_fold4_V_drop103.csv", index=False)

sample_submission5.to_csv("./submit_xgb_fold5_V_drop103.csv", index=False)

sample_submission6.to_csv("./submit_xgb_fold6_V_drop103.csv", index=False)

sample_submission7.to_csv("./submit_xgb_fold7_V_drop103.csv", index=False)

sample_submission8.to_csv("./submit_xgb_fold8_V_drop103.csv", index=False)

sample_submission9.to_csv("./submit_xgb_fold9_V_drop103.csv", index=False)

sample_submission10.to_csv("./submit_xgb_fold10_V_drop103.csv", index=False)

df_imp = pd.DataFrame({'imp':XGB.feature_importances_}, index = XGB.feature_names_in_)
df_imp = df_imp[df_imp.imp > 0].sort_values('imp').copy()
print(df_imp)
