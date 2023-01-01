import pandas as pd

train_x = pd.read_csv('./train_x_309plus3.csv')
train_y = pd.read_csv('./train_y_309plus3.csv')
test_x = pd.read_csv('./test_x_309plus3.csv')


#train = pd.concat([train_x,train_y], axis = 1)

#print(train.info())

#추가

from tqdm import tqdm
from time import sleep
from numba import jit
@jit(nopython=True)

def cluster1(train_x):   
    train_x['cluster'] = -1
    for i in tqdm(range(len(train_x))):
      sum = [0,0,0,0,0,0,0,0,0,0]
      lon = train_x['start_latitude'][i] - 33.46118983
      lat = train_x['start_longitude'][i] - 126.42620271
      sum[2] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.39605554
      lat = train_x['start_longitude'][i] - 126.78753828
      sum[6] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.2584619
      lat = train_x['start_longitude'][i] - 126.50091375
      sum[9] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.38950759
      lat = train_x['start_longitude'][i] - 126.26054059
      sum[1] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.49810998
      lat = train_x['start_longitude'][i] - 126.54241199
      sum[3] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.27122724
      lat = train_x['start_longitude'][i] - 126.3897301
      sum[0] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.25962238
      lat = train_x['start_longitude'][i] - 126.56981339
      sum[8] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.31407046
      lat = train_x['start_longitude'][i] - 126.67560911
      sum[7] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.48173776
      lat = train_x['start_longitude'][i] - 126.88263094
      sum[5] = abs(lon) + abs(lat) 
      lon = train_x['start_latitude'][i] - 33.47104482
      lat = train_x['start_longitude'][i] - 126.69386631
      sum[4] = abs(lon) + abs(lat) 
      numbering = 0
      for j in range(len(sum)):
        if(min(sum) == sum[j]):
          train_x['cluster'][i] = numbering
        numbering += 1
      sleep(3)
    return train_x

train_x = cluster1(train_x)


def cluster2(test_x):
    
    test_x['cluster'] = -1
    for i in tqdm(range(len(test_x))):
      sum = [0,0,0,0,0,0,0,0,0,0]
      lon = test_x['start_latitude'][i] - 33.46118983
      lat = test_x['start_longitude'][i] - 126.42620271
      sum[2] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.39605554
      lat = test_x['start_longitude'][i] - 126.78753828
      sum[6] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.2584619
      lat = test_x['start_longitude'][i] - 126.50091375
      sum[9] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.38950759
      lat = test_x['start_longitude'][i] - 126.26054059
      sum[1] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.49810998
      lat = test_x['start_longitude'][i] - 126.54241199
      sum[3] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.27122724
      lat = test_x['start_longitude'][i] - 126.3897301
      sum[0] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.25962238
      lat = test_x['start_longitude'][i] - 126.56981339
      sum[8] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.31407046
      lat = test_x['start_longitude'][i] - 126.67560911
      sum[7] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.48173776
      lat = test_x['start_longitude'][i] - 126.88263094
      sum[5] = abs(lon) + abs(lat) 
      lon = test_x['start_latitude'][i] - 33.47104482
      lat = test_x['start_longitude'][i] - 126.69386631
      sum[4] = abs(lon) + abs(lat) 
      numbering = 0
      for j in range(len(sum)):
        if(min(sum) == sum[j]):
          test_x['cluster'][i] = numbering
        numbering += 1
      sleep(3)
    return test_x

test_x = cluster2(test_x)


train_x.to_csv("train_x_309plus3_clus.csv",index = False)
train_y.to_csv("train_y_309plus3_clus.csv",index = False)
test_x.to_csv("test_x_309plus3_clus.csv",index = False)
