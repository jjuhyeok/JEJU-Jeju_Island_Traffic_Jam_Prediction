import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.model_selection import StratifiedKFold
import math
from IPython.display import display
from sklearn.decomposition import PCA
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

train_x = pd.read_csv('./train_x_309plus3_2.csv')
train_y = pd.read_csv('./train_y_309plus3_2.csv')
test_x = pd.read_csv('./test_x_309plus3_2.csv')

#다포함시키면 9번 돌았을때 2.98
#'location_cluster_41', '평균속도_mean', '평균속도_median','평균속도_mean_log', '평균속도_median_log', 
############################Test2#############################################
coord = train_x[['start_latitude','start_longitude']]
coord_t = test_x[['start_latitude','start_longitude']]

pca = PCA(n_components=2)
pca.fit(coord)

coord_pca = pca.fit_transform(coord)
coord_pca_t = pca.transform(coord_t)

train_x['coord_pca1_s'] = coord_pca[:, 0]
train_x['coord_pca2_s'] = coord_pca[:, 1]

test_x['coord_pca1_s'] = coord_pca_t[:, 0]
test_x['coord_pca2_s'] = coord_pca_t[:, 1]


coord = train_x[['end_latitude','end_longitude']]
coord_t = test_x[['end_latitude','end_longitude']]

pca = PCA(n_components=2)
pca.fit(coord)

coord_pca = pca.fit_transform(coord)
coord_pca_t = pca.transform(coord_t)

train_x['coord_pca1_e'] = coord_pca[:, 0]
train_x['coord_pca2_e'] = coord_pca[:, 1]

test_x['coord_pca1_e'] = coord_pca_t[:, 0]
test_x['coord_pca2_e'] = coord_pca_t[:, 1]




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

train_x, test_x = make_cluster_41(train_x, test_x)

from sklearn.preprocessing import OneHotEncoder

train_x['location_41cluster'] = train_x['location_41cluster'].astype(str)
test_x['location_41cluster'] = test_x['location_41cluster'].astype(str)

ohe = OneHotEncoder(sparse=False)
train_cat = ohe.fit_transform(train_x[['location_41cluster']])
print(ohe.categories_[0])
pd.DataFrame(train_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])
train_x = pd.concat([train_x.drop(['location_41cluster'], axis = 1), pd.DataFrame(train_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])], axis=1)


# fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
test_cat = ohe.transform(test_x[['location_41cluster']])
pd.DataFrame(test_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])
test_x = pd.concat([test_x.drop(['location_41cluster'], axis = 1), pd.DataFrame(test_cat, columns=['location_41cluster' + col for col in ohe.categories_[0]])], axis=1)




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
            'random_state' : 9555,
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

#V2+41
cat = CatBoostRegressor('iterations': 497, 'learning_rate': 0.34343323337071235, 'depth': 15, 'min_data_in_leaf': 30, 'reg_lambda': 30.955594650411673, 'subsample': 0.7828258614477996, 'random_strength': 99.44802647564642, 'od_wait': 50, 'leaf_estimation_iterations': 15, 'bagging_temperature': 18.664732809389324, 'colsample_bylevel': 0.9540724750341851)



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

cat_models={}

for fold in range(5):
  print(f'===================================={fold+1}============================================')
  train_idx, valid_idx = folds[fold]
  X_train = train_x.iloc[train_idx].values 
  X_valid = train_x.iloc[valid_idx].values
  y_train = train_y['target'][train_idx].values
  y_valid = train_y['target'][valid_idx].values

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

'''
