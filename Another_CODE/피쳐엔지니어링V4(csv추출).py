import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc

from IPython.display import display

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')


csv_to_parquet('./train.csv', 'train')
csv_to_parquet('./test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')


##############Feature Engineering#######################

str_col = ['start_turn_restricted','end_turn_restricted']
for i in str_col:
    le = LabelEncoder()
    le=le.fit(train[i])
    train[i]=le.transform(train[i])
    
    for label in np.unique(test[i]):
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test[i]=le.transform(test[i])

train_y = train['target'] 

train_x = train.drop(['id', 'target', 'start_node_name', 'end_node_name','vehicle_restricted','road_rating','height_restricted'], axis=1)

test_x = test.drop(['id',  'start_node_name', 'end_node_name','vehicle_restricted','road_rating','height_restricted'], axis=1)

#위경도 scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_x['start_latitude_MinMax'] = scaler.fit_transform(train_x['start_latitude'].values.reshape(-1,1))
test_x['start_latitude_MinMax'] = scaler.transform(test_x['start_latitude'].values.reshape(-1,1))

train_x['start_longitude_MinMax'] = scaler.fit_transform(train_x['start_longitude'].values.reshape(-1,1))
test_x['start_longitude_MinMax'] = scaler.transform(test_x['start_longitude'].values.reshape(-1,1))

train_x['end_latitude_MinMax'] = scaler.fit_transform(train_x['end_latitude'].values.reshape(-1,1))
test_x['end_latitude_MinMax'] = scaler.transform(test_x['end_latitude'].values.reshape(-1,1))

train_x['end_longitude_MinMax'] = scaler.fit_transform(train_x['end_longitude'].values.reshape(-1,1))
test_x['end_longitude_MinMax'] = scaler.transform(test_x['end_longitude'].values.reshape(-1,1))


#위경도 scaling
train_x['start_latitude'] = train_x['start_latitude'] - 33
train_x['end_latitude'] = train_x['end_latitude'] - 33
train_x['start_longitude'] = train_x['start_longitude'] - 126
train_x['end_longitude'] = train_x['end_longitude'] - 126

test_x['start_latitude'] = test_x['start_latitude'] - 33
test_x['end_latitude'] = test_x['end_latitude'] - 33
test_x['start_longitude'] = test_x['start_longitude'] - 126
test_x['end_longitude'] = test_x['end_longitude'] - 126


#로그변환
import math




#df = df.replace({'열 이름' : 기존 값}, 변경 값)
#train_x.replace({'maximum_speed_limit' : 0}, 1)
#df.loc[df['A'] == 9999, 'B'] = -500
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] <= 0)] = 1
train_x['weight_restricted'][(train_x['weight_restricted'] <= 0)] = 1
#train_x['height_restricted'][(train_x['height_restricted'] <= 0)] = 1
train_x['start_latitude'][(train_x['start_latitude'] <= 0)] = 1
train_x['end_latitude'][(train_x['end_latitude'] <= 0)] = 1
train_x['start_longitude'][(train_x['start_longitude'] <= 0)] = 1
train_x['end_longitude'][(train_x['end_longitude'] <= 0)] = 1

#train_x['start_latitude_MinMax'][(train_x['start_latitude_MinMax'] <= 0)] = 1
#train_x['start_longitude_MinMax'][(train_x['start_longitude_MinMax'] <= 0)] = 1
#train_x['end_latitude_MinMax'][(train_x['end_latitude_MinMax'] <= 0)] = 1
#train_x['end_longitude_MinMax'][(train_x['end_longitude_MinMax'] <= 0)] = 1

test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] <= 0)] = 1
test_x['weight_restricted'][(test_x['weight_restricted'] <= 0)] = 1
#test_x['height_restricted'][(test_x['height_restricted'] <= 0)] = 1
test_x['start_latitude'][(test_x['start_latitude'] <= 0)] = 1
test_x['end_latitude'][(test_x['end_latitude'] <= 0)] = 1
test_x['start_longitude'][(test_x['start_longitude'] <= 0)] = 1
test_x['end_longitude'][(test_x['end_longitude'] <= 0)] = 1

#test_x['start_latitude_MinMax'][(test_x['start_latitude_MinMax'] <= 0)] = 1
#test_x['start_longitude_MinMax'][(test_x['start_longitude_MinMax'] <= 0)] = 1
#test_x['end_latitude_MinMax'][(test_x['end_latitude_MinMax'] <= 0)] = 1
#test_x['end_longitude_MinMax'][(test_x['end_longitude_MinMax'] <= 0)] = 1

#train_x.replace({'height_restricted' : 0}, 1)
#train_x.replace({'start_latitude' : 0}, 1)
#train_x.replace({'end_latitude' : 0}, 1)
#train_x.replace({'start_longitude' : 0}, 1)
#train_x.replace({'end_longitude' : 0}, 1)


#로그변환
basic_col = train_x.columns
for col in basic_col:
    if col in ['day_of_week','road_name','base_hour','road_in_use','lane_count', 'road_rating','multi_linked',"connect_code", 'road_type', 'start_turn_restricted', 'end_turn_restricted','start_latitude_MinMax','start_longitude_MinMax','end_latitude_MinMax','end_longitude_MinMax']:
        continue
    print(col)
    train_x[col+"_log"] = train_x[col].apply(lambda x : math.log(x))

basic_col_t = test_x.columns
for col in basic_col_t:
    if col in ['day_of_week','road_name','base_hour','road_in_use','lane_count', 'road_rating', 'multi_linked',"connect_code", 'road_type', 'start_turn_restricted', 'end_turn_restricted','start_latitude_MinMax','start_longitude_MinMax','end_latitude_MinMax','end_longitude_MinMax']:
        continue
    test_x[col+"_log"] = test_x[col].apply(lambda x : math.log(x))




train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 30)] = 'f'
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 40)] = 'e'
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 50)] = 'd'
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 60)] = 'c'
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 70)] = 'b'
train_x['maximum_speed_limit'][(train_x['maximum_speed_limit'] == 80)] = 'a'

test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 30)] = 'f'
test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 40)] = 'e'
test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 50)] = 'd'
test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 60)] = 'c'
test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 70)] = 'b'
test_x['maximum_speed_limit'][(test_x['maximum_speed_limit'] == 80)] = 'a'


train_x = pd.get_dummies(data = train_x, columns = ['maximum_speed_limit'], prefix = 'maximum_speed_limit_')
#train_x = train_x.drop(['maximum_speed_limit'], axis = 1, inplace = False)
test_x = pd.get_dummies(data = test_x, columns = ['maximum_speed_limit'], prefix = 'maximum_speed_limit_')

#제한이 클 수록 차들 평균 통행 속도가 빠른 곳일 것
train_x['weight_restricted'][(train_x['weight_restricted'] == 0)] = 'd'
train_x['weight_restricted'][(train_x['weight_restricted'] == 32400)] = 'c'
train_x['weight_restricted'][(train_x['weight_restricted'] == 43200)] = 'b'
train_x['weight_restricted'][(train_x['weight_restricted'] == 50000)] = 'a'


test_x['weight_restricted'][(test_x['weight_restricted'] == 0)] = 'd'
test_x['weight_restricted'][(test_x['weight_restricted'] == 32400)] = 'c'
test_x['weight_restricted'][(test_x['weight_restricted'] == 43200)] = 'b'
test_x['weight_restricted'][(test_x['weight_restricted'] == 50000)] = 'a'



train_x = pd.get_dummies(data = train_x, columns = ['weight_restricted'], prefix = 'maximum_speed_limit_')
#train_x = train_x.drop(['maximum_speed_limit'], axis = 1, inplace = False)
test_x = pd.get_dummies(data = test_x, columns = ['weight_restricted'], prefix = 'maximum_speed_limit_')
#다른 조건일때도 해줘야되는데 막 제한이 70000인게 있을수도있는데



train_x['connect_code'][(train_x['connect_code'] == 103)] = 1

test_x['connect_code'][(test_x['connect_code'] == 103)] = 1




#base_date 나눠주고 드랍

train_x["year"]= (train_x["base_date"] // 10000) - 2021
train_x['month'] = (train_x["base_date"] % 10000) // 100
train_x['season'] = 0
train_x['season'][(train_x['month'] == 12)] = 'a'
train_x['season'][(train_x['month'] == 1)] = 'a'
train_x['season'][(train_x['month'] == 2)] = 'a'
train_x['season'][(train_x['month'] == 3)] = 'b'
train_x['season'][(train_x['month'] == 4)] = 'b'
train_x['season'][(train_x['month'] == 5)] = 'b'
train_x['season'][(train_x['month'] == 6)] = 'c'
train_x['season'][(train_x['month'] == 7)] = 'c'
train_x['season'][(train_x['month'] == 8)] = 'c'
train_x['season'][(train_x['month'] == 9)] = 'd'
train_x['season'][(train_x['month'] == 10)] = 'd'
train_x['season'][(train_x['month'] == 11)] = 'd'
train_x.drop(['month'], axis = 1 , inplace = True)

#train_x['day'] = (train_x["base_date"] % 100)

train_x['month_day'] = (train_x["base_date"] % 10000) #공휴일

train_x.drop(['base_date'], axis = 1 , inplace = True)


test_x["year"]= (test_x["base_date"] // 10000) - 2021
test_x['month'] = (test_x["base_date"] % 10000) // 100 
test_x['season'] = 0
test_x['season'][(test_x['month'] == 12)] = 'a'
test_x['season'][(test_x['month'] == 1)] = 'a'
test_x['season'][(test_x['month'] == 2)] = 'a'
test_x['season'][(test_x['month'] == 3)] = 'b'
test_x['season'][(test_x['month'] == 4)] = 'b'
test_x['season'][(test_x['month'] == 5)] = 'b'
test_x['season'][(test_x['month'] == 6)] = 'c'
test_x['season'][(test_x['month'] == 7)] = 'c'
test_x['season'][(test_x['month'] == 8)] = 'c'
test_x['season'][(test_x['month'] == 9)] = 'd'
test_x['season'][(test_x['month'] == 10)] = 'd'
test_x['season'][(test_x['month'] == 11)] = 'd'
test_x.drop(['month'], axis = 1 , inplace = True)

#test_x['day'] = (test_x["base_date"] % 100)

test_x['month_day'] = (test_x["base_date"] % 10000) #공휴일

test_x.drop(['base_date'], axis = 1 , inplace = True)
#train_x['year'].unique()

train_x = pd.get_dummies(data = train_x, columns = ['season'], prefix = 'season_')
test_x = pd.get_dummies(data = test_x, columns = ['season'], prefix = 'season_')



#날짜가 정해진 공유일에는 교통난 심할것
train_x['month_day'][(train_x['month_day'] == 101)] = 'b'
train_x['month_day'][(train_x['month_day'] == 301)] = 'b'
train_x['month_day'][(train_x['month_day'] == 505)] = 'b'
train_x['month_day'][(train_x['month_day'] == 606)] = 'b'
train_x['month_day'][(train_x['month_day'] == 815)] = 'b'
train_x['month_day'][(train_x['month_day'] == 1003)] = 'b'
train_x['month_day'][(train_x['month_day'] == 1009)] = 'b'
train_x['month_day'][(train_x['month_day'] == 1225)] = 'b'
train_x['month_day'][(train_x['month_day'] == 1224)] = 'b'

train_x['month_day'][(train_x['month_day'] != 1)] = 'a'

test_x['month_day'][(test_x['month_day'] == 101)] = 'b'
test_x['month_day'][(test_x['month_day'] == 301)] = 'b'
test_x['month_day'][(test_x['month_day'] == 505)] = 'b'
test_x['month_day'][(test_x['month_day'] == 606)] = 'b'
test_x['month_day'][(test_x['month_day'] == 815)] = 'b'
test_x['month_day'][(test_x['month_day'] == 1003)] = 'b'
test_x['month_day'][(test_x['month_day'] == 1009)] = 'b'
test_x['month_day'][(test_x['month_day'] == 1225)] = 'b'
test_x['month_day'][(test_x['month_day'] == 1224)] = 'b'

test_x['month_day'][(test_x['month_day'] != 1)] = 'a'

train_x = pd.get_dummies(data = train_x, columns = ['month_day'], prefix = 'month_day_')
test_x = pd.get_dummies(data = test_x, columns = ['month_day'], prefix = 'month_day_')





#금토는 많이 막힐것
train_x['day_of_week'][(train_x['day_of_week'] == '월')] = 'b'
train_x['day_of_week'][(train_x['day_of_week'] == '화')] = 'a'
train_x['day_of_week'][(train_x['day_of_week'] == '수')] = 'a'
train_x['day_of_week'][(train_x['day_of_week'] == '목')] = 'a'
train_x['day_of_week'][(train_x['day_of_week'] == '금')] = 'c'
train_x['day_of_week'][(train_x['day_of_week'] == '토')] = 'c'
train_x['day_of_week'][(train_x['day_of_week'] == '일')] = 'b'



test_x['day_of_week'][(test_x['day_of_week'] == '월')] = 'b'
test_x['day_of_week'][(test_x['day_of_week'] == '화')] = 'a'
test_x['day_of_week'][(test_x['day_of_week'] == '수')] = 'a'
test_x['day_of_week'][(test_x['day_of_week'] == '목')] = 'a'
test_x['day_of_week'][(test_x['day_of_week'] == '금')] = 'c'
test_x['day_of_week'][(test_x['day_of_week'] == '토')] = 'c'
test_x['day_of_week'][(test_x['day_of_week'] == '일')] = 'b'


#train_x['day_of_week'].unique()
train_x = pd.get_dummies(data = train_x, columns = ['day_of_week'], prefix = 'day_of_week_')
test_x = pd.get_dummies(data = test_x, columns = ['day_of_week'], prefix = 'day_of_week_')


train_x['lazy_hour'] = 0
train_x['lazy_hour'][(train_x['base_hour'] == 0)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 1)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 2)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 3)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 4)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 5)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 6)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 7)] = 'c'
train_x['lazy_hour'][(train_x['base_hour'] == 8)] = 'd'
train_x['lazy_hour'][(train_x['base_hour'] == 9)] = 'c'
train_x['lazy_hour'][(train_x['base_hour'] == 10)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 11)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 12)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 13)] = 'c'
train_x['lazy_hour'][(train_x['base_hour'] == 14)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 15)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 16)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 17)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 18)] = 'c'
train_x['lazy_hour'][(train_x['base_hour'] == 19)] = 'd'
train_x['lazy_hour'][(train_x['base_hour'] == 20)] = 'c'
train_x['lazy_hour'][(train_x['base_hour'] == 21)] = 'b'
train_x['lazy_hour'][(train_x['base_hour'] == 22)] = 'a'
train_x['lazy_hour'][(train_x['base_hour'] == 23)] = 'a'



test_x['lazy_hour'] = 0
test_x['lazy_hour'][(test_x['base_hour'] == 0)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 1)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 2)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 3)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 4)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 5)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 6)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 7)] = 'c'
test_x['lazy_hour'][(test_x['base_hour'] == 8)] = 'd'
test_x['lazy_hour'][(test_x['base_hour'] == 9)] = 'c'
test_x['lazy_hour'][(test_x['base_hour'] == 10)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 11)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 12)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 13)] = 'c'
test_x['lazy_hour'][(test_x['base_hour'] == 14)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 15)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 16)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 17)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 18)] = 'c'
test_x['lazy_hour'][(test_x['base_hour'] == 19)] = 'd'
test_x['lazy_hour'][(test_x['base_hour'] == 20)] = 'c'
test_x['lazy_hour'][(test_x['base_hour'] == 21)] = 'b'
test_x['lazy_hour'][(test_x['base_hour'] == 22)] = 'a'
test_x['lazy_hour'][(test_x['base_hour'] == 23)] = 'a'



train_x = pd.get_dummies(data = train_x, columns = ['lazy_hour'], prefix = 'lazy_hour_')
test_x = pd.get_dummies(data = test_x, columns = ['lazy_hour'], prefix = 'lazy_hour_')




train_x['road_name'][(train_x['road_type'] == 3)] = '국_지_도'

test_x['road_name'][(test_x['road_type'] == 3)] = '국_지_도'


#train_x['road_name'].apply(lambda x : '0' if '국도' in x)
#교 도 3으로 들어감
train_x['rname_new'] = 0
train_x['rname_new'][train_x['road_name'].str.contains('국도')] = 'a'
train_x['rname_new'][train_x['road_name'].str.contains('지방도')] = 'b'
train_x['rname_new'][train_x['road_name'].str.contains('로')] = 'c'
train_x['rname_new'][train_x['road_name'].str.contains('교')] = 'd'
train_x['rname_new'][train_x['road_name'].str.contains('국_지_도')] = 'b'
train_x['rname_new'][train_x['road_name'].str.contains('NaN')] = 'c'


test_x['rname_new'] = 0
test_x['rname_new'][test_x['road_name'].str.contains('국도')] = 'a'
test_x['rname_new'][test_x['road_name'].str.contains('지방도')] = 'b'
test_x['rname_new'][test_x['road_name'].str.contains('로')] = 'c'
test_x['rname_new'][test_x['road_name'].str.contains('교')] = 'd'
test_x['rname_new'][test_x['road_name'].str.contains('국_지_도')] = 'b'
test_x['rname_new'][test_x['road_name'].str.contains('NaN')] = 'c'


train_x = pd.get_dummies(data = train_x, columns = ['rname_new'], prefix = 'rname_new')
test_x = pd.get_dummies(data = test_x, columns = ['rname_new'], prefix = 'rname_new')

train_x.drop(['road_name'],axis = 1, inplace = True)
test_x.drop(['road_name'],axis = 1, inplace = True)


train_x.drop(['base_hour'],axis = 1 , inplace = True)
test_x.drop(['base_hour'],axis = 1 , inplace = True)



train_x.to_csv("EngineeringV4_train_x.csv", index = False)
train_y.to_csv("EngineeringV4_train_y.csv", index = False)
test_x.to_csv("EngineeringV4_test_x.csv", index = False)


print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")
print("====================Finish Feature Engineering======================")

'''
csv_to_parquet('./train.csv', 'train')
csv_to_parquet('./test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
'''
#######################Modeling########################

from pycaret.regression import *
caret = pd.concat([train_x,train_y], axis = 1)



model = setup(caret, target = 'target',  session_id = 42, silent = True, use_gpu = True, fold = 5, html=False)




#best_3 = compare_models(include = [ 'dt', 'lightgbm' , 'catboost', 'xgboost'], n_select = 3, fold = 5, sort = 'MAE')
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")
print("==================best_3_complete========================")

total_models = []

model_dt = create_model('dt', fold = 5)
model_xgboost = create_model('xgboost', fold = 5)
#model_catboost = create_model('catboost', fold = 5)
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
#total_models.append(model_dt_tune)


'''
#tuned_best_3 = [tune_model(i, optimize = 'MAE', fold = 5, n_iter = 3, choose_better = False, verbose = True) for i in best_3]
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")
print("==================tuned_best_3_complete========================")'''
lightgbm = create_model('lightgbm')
stacker = stack_models(total_models,fold = 5, meta_model = lightgbm) #튜닝안한상태
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
y_predict = predict_model(final_model, data=test_x)


#######################submission########################
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['target'] = y_predict['Label']
sample_submission.to_csv("./submit2.csv", index = False)

print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")
print("==================finish========================")


#save_model
save_model(final_model, 'my_best_pipeline_tuned', verbose=True)

#loaded_model = load_model('my_best_pipeline')
