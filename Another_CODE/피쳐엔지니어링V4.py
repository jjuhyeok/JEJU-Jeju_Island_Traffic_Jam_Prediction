import pandas as pd
import numpy as np
cluster_centers = np.array([[33.2799141, 126.7207056], [33.2482109, 126.5113278], [33.2268151, 126.2522254], [33.2507866, 126.477585], [33.2578824, 126.5689267], [33.2558577, 126.5602063], [33.3810625, 126.8767558], [37.691055, 126.753565], [33.2496801, 126.3371482], [33.2687729, 126.5867363], [33.2542814, 126.3979513], [33.2458588, 126.5652696], [33.2539386, 126.4339728], [37.4841833, 126.9497246], [33.2477973, 126.5613541], [33.326494, 126.831638], [33.3099343, 126.7674357], [33.5170777, 126.5445896], [33.5225711, 126.8520445], [33.4830731, 126.4772131], [33.5028456, 126.4683154], [33.4916823, 126.5947483], [33.503187, 126.519751], [33.5116788, 126.5222179], [33.521762, 126.5855263], [33.4762704, 126.5450898], [33.4619478, 126.3295244], [33.4881534, 126.4969519], [33.495125, 126.5116449], [33.4928462, 126.4322356], [33.5096339, 126.514017], [33.5114842, 126.5117556], [33.5067302, 126.5270745], [33.4969863, 126.5353375], [33.4997903, 126.4582053], [33.5150396, 126.526393], [33.5116235, 126.5383642], [33.5344076, 126.6342455], [33.3501173, 126.184217], [33.4103985, 126.2671535], [33.5202063, 126.5655282]])

train_x = pd.read_csv('./train_x_309plus3_2.csv')
train_y = pd.read_csv('train_y_309plus3_2.csv')
test_x = pd.read_csv('./test_x_309plus3_2.csv')
train_x = pd.concat([train_x, train_y], axis = 1)
def make_cluster_41(train, test):
    from sklearn.cluster import KMeans
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    k_mean = KMeans(n_clusters=41, init=cluster_centers, random_state=np.random.RandomState(seed=14))
    #k_mean.fit(train_c)
    train['location_cluster_41'] = k_mean.fit_predict(train_c)
    test['location_cluster_41'] = k_mean.predict(test_c)
    return train, test

train_x, test_x = make_cluster_41(train_x, test_x)


addlist=['남원읍', '대륜동', '대정읍', '대천동', '동홍동', '서홍동', '성산읍', '송산동', '안덕면',
       '영천동', '예래동', '정방동', '중문동', '중앙동', '천지동', '표선면', '효돈동', '건입동',
       '구좌읍', '노형동', '도두동', '봉개동', '삼도1동', '삼도2동', '삼양동', '아라동', '애월읍',
       '연동', '오라동', '외도동', '용담1동', '용담2동', '이도1동', '이도2동', '이호동', '일도1동',
       '일도2동', '조천읍', '한경면', '한림읍', '화북동']
def adr(df):
  df['읍면동명']='알수없음'
  for i in range(0,len(addlist)):
    df['읍면동명'][(df['location_cluster_41'] == i)] = addlist[i] 

adr(train_x)
adr(test_x)

print(train_x.info())
print(test_x.info())



jeju = pd.read_csv('./add_data.csv', encoding = 'cp949')

print(jeju.info())
train_x = pd.merge(train_x, jeju, on = '읍면동명')
test_x = pd.merge(test_x, jeju, on = '읍면동명')

train_x.drop(['읍면동명'], axis = 1 , inplace = True)
test_x.drop(['읍면동명'], axis = 1, inplace = True)
jeju.drop(['읍면동명'], axis = 1 , inplace = True)

import math

basic_col = jeju.columns
for col in basic_col:
    print(col)
    train_x[col+"_log"] = train_x[col].apply(lambda x : math.log(x+1))

basic_col_t = jeju.columns
for col in basic_col_t:
    test_x[col+"_log"] = test_x[col].apply(lambda x : math.log(x+1))



train_y['target'] = train_x['target']

train_x.drop(['target'], axis = 1, inplace = True)
train_x.to_csv('train_x_V4.csv', index = False)


train_y.to_csv('train_y_V4.csv', index = False)
test_x.to_csv('test_x_V4.csv', index = False)
