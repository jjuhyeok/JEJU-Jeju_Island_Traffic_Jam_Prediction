import pandas as pd

train_x = pd.read_csv('./train_x_309plus____103test.csv')
train_y = pd.read_csv('./train_y_309plus____103test.csv')
test_x = pd.read_csv('./test_x_309plus____103test.csv')


#train = pd.concat([train_x,train_y], axis = 1)

#print(train.info())

#추가

from tqdm import tqdm
from time import sleep
'''
train['cluster'] = -1
for i in tqdm(range(len(train_x))):
  sum = [0,0,0,0,0,0,0,0,0,0]
  lon = train_x.iloc[i,12] - 33.46118983
  lat = train_x.iloc[i,13] - 126.42620271
  sum[2] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.39605554
  lat = train_x.iloc[i,13] - 126.78753828
  sum[6] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.2584619
  lat = train_x.iloc[i,13] - 126.50091375
  sum[9] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.38950759
  lat = train_x.iloc[i,13] - 126.26054059
  sum[1] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.49810998
  lat = train_x.iloc[i,13] - 126.54241199
  sum[3] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.27122724
  lat = train_x.iloc[i,13] - 126.3897301
  sum[0] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.25962238
  lat = train_x.iloc[i,13] - 126.56981339
  sum[8] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.31407046
  lat = train_x.iloc[i,13] - 126.67560911
  sum[7] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.48173776
  lat = train_x.iloc[i,13] - 126.88263094
  sum[5] = abs(lon) + abs(lat) 
  lon = train_x.iloc[i,12] - 33.47104482
  lat = train_x.iloc[i,13] - 126.69386631
  sum[4] = abs(lon) + abs(lat) 
  sum2 = pd.DataFrame(sum)
  numbering = 0
  for j in range(len(sum)):
    if(sum2[0].min() == sum[j]):
      train_x.iloc[i,:-1] = numbering
    numbering += 1
  sleep(0.1)

test_x['cluster'] = -1
for i in tqdm(range(len(test_x))):
  sum = [0,0,0,0,0,0,0,0,0,0]
  lon = test_x.iloc[i,12] - 33.46118983
  lat = test_x.iloc[i,13] - 126.42620271
  sum[2] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.39605554
  lat = test_x.iloc[i,13] - 126.78753828
  sum[6] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.2584619
  lat = test_x.iloc[i,13] - 126.50091375
  sum[9] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.38950759
  lat = test_x.iloc[i,13] - 126.26054059
  sum[1] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.49810998
  lat = test_x.iloc[i,13] - 126.54241199
  sum[3] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.27122724
  lat = test_x.iloc[i,13] - 126.3897301
  sum[0] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.25962238
  lat = test_x.iloc[i,13] - 126.56981339
  sum[8] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.31407046
  lat = test_x.iloc[i,13] - 126.67560911
  sum[7] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.48173776
  lat = test_x.iloc[i,13] - 126.88263094
  sum[5] = abs(lon) + abs(lat) 
  lon = test_x.iloc[i,12] - 33.47104482
  lat = test_x.iloc[i,13] - 126.69386631
  sum[4] = abs(lon) + abs(lat) 
  sum3 = pd.DataFrame(sum)
  numbering = 0
  for j in range(len(sum)):
    if(sum3[0].min() == sum[j]):
      test_x.iloc[i,:-1] = numbering
    numbering += 1
  sleep(0.1)


train_x.to_csv("train_x_309plus3_back.csv",index = False)
train_y.to_csv("train_y_309plus3_back.csv",index = False)
test_x.to_csv("test_x_309plus3_back.csv",index = False)
'''

'''
def make_cluster_100(train, test):
    from sklearn.cluster import KMeans
    train_c = train[['start_latitude', 'start_longitude']]
    test_c = test[['start_latitude', 'start_longitude']]
    
    k_mean = KMeans(n_clusters=100, init='k-means++')
    train['location_cluster_50'] = k_mean.fit_predict(train_c)
    test['location_cluster_50'] = k_mean.predict(test_c)
    
    return train, test

train_x, test_x = make_cluster_100(train_x, test_x)
'''

#거리
def hanra_dist(df):
    from haversine import haversine
    jeju_location = (33.361417, 126.529417)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

train_x['hanra_dist'] = hanra_dist(train_x)
test_x['hanra_dist'] = hanra_dist(test_x)


def sungsan_dist(df):
    from haversine import haversine
    jeju_location = (33.458528, 126.94225)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

train_x['sungsan_dist'] = sungsan_dist(train_x)
test_x['sungsan_dist'] = sungsan_dist(test_x)


def joongmoon_dist(df):
    from haversine import haversine
    jeju_location = (33.246340915095914, 126.41973291093717)
    end_location = tuple(zip(df['end_latitude'], df['end_longitude']))
    hsine = [haversine(i, jeju_location) for i in end_location]

    return hsine

train_x['joongmoon_dist'] = sungsan_dist(train_x)
test_x['joongmoon_dist'] = sungsan_dist(test_x)


def speed_time(train, test, col, col_name):
    speed = train.groupby([col, 'base_hour'])['target'].agg([(col_name, 'mean')]).reset_index()
    train = pd.merge(train, speed, on=[col, 'base_hour'], how='left')
    test = pd.merge(test, speed, on=[col, 'base_hour'], how='left')
    return train, test

train_x = pd.concat([train_x,train_y],axis=1)
train_x, test_x = speed_time(train_x,test_x,'road_name','section_speed_t')
train_x, test_x = speed_time(train_x,test_x,'road_name','start_speed_t')
train_x, test_x = speed_time(train_x,test_x,'road_name','end_speed_t')
train_x.drop(['target'], axis = 1, inplace = True)


import math

basic_col = ['section_speed_t','start_speed_t','end_speed_t', 'section_speed','start_speed','end_speed']
for col in basic_col:
    print(col)
    train_x[col+"_log"] = train_x[col].apply(lambda x : math.log(x))

basic_col_t = ['section_speed_t','start_speed_t','end_speed_t', 'section_speed','start_speed','end_speed']
for col in basic_col_t:
    test_x[col+"_log"] = test_x[col].apply(lambda x : math.log(x))




train_x['location_dum'] = 0
train_x['location_dum'][(train_x['location_cluster'] == 0)] = 'a'
train_x['location_dum'][(train_x['location_cluster'] == 1)] = 'b'
train_x['location_dum'][(train_x['location_cluster'] == 2)] = 'c'
train_x['location_dum'][(train_x['location_cluster'] == 3)] = 'd'

test_x['location_dum'] = 0
test_x['location_dum'][(test_x['location_cluster'] == 0)] = 'a'
test_x['location_dum'][(test_x['location_cluster'] == 1)] = 'b'
test_x['location_dum'][(test_x['location_cluster'] == 2)] = 'c'
test_x['location_dum'][(test_x['location_cluster'] == 3)] = 'd'



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

train_cat = ohe.fit_transform(train_x[['location_dum']])
pd.DataFrame(train_cat, columns=['location_dum' + col for col in ohe.categories_[0]])
train_x = pd.concat([train_x.drop(['location_dum'], axis = 1), pd.DataFrame(train_cat, columns=['location_dum' + col for col in ohe.categories_[0]])], axis=1)


# fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
test_cat = ohe.transform(test_x[['location_dum']])
pd.DataFrame(test_cat, columns=['location_dum' + col for col in ohe.categories_[0]])
test_x = pd.concat([test_x.drop(['location_dum'], axis = 1), pd.DataFrame(test_cat, columns=['location_dum' + col for col in ohe.categories_[0]])], axis=1)







print(train_x)

print(train_x.info())
print(test_x.info())

train_x.to_csv("train_x_309plus3_2____103test.csv",index = False)
train_y.to_csv("train_y_309plus3_2____103test.csv",index = False)
test_x.to_csv("test_x_309plus3_2____103test.csv",index = False)
