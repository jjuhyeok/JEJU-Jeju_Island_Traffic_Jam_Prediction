import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font',family='NanumBarunGothic')
train_x = pd.read_csv('./train_x_309plus3_2.csv')
train_y = pd.read_csv('./train_y_309plus3_2.csv')
test_x = pd.read_csv('./test_x_309plus3_2.csv')
train = pd.concat([train_x, train_y], axis = 1)

df = train

fig = plt.figure(figsize = (16, 12))

ax1 = plt.subplot(221)
sns.scatterplot(x='start_longitude', y='start_latitude', hue='location_duma', size='target', sizes = (1,300), data=df, ax=ax1)

ax2 = plt.subplot(222)
sns.scatterplot(x='start_longitude', y='start_latitude', hue='location_dumb', size='target', sizes = (1,300), data=df, ax=ax2)

ax3 = plt.subplot(223)
sns.scatterplot(x='start_longitude', y='start_latitude', hue='location_dumc', size='target', sizes = (1,300), data=df, ax=ax3)

ax4 = plt.subplot(224)
sns.scatterplot(x='start_longitude', y='start_latitude', hue='location_dumd', size='target', sizes = (1,300), data=df, ax=ax4)

plt.show()


'''
# pca for lat, long
from sklearn.decomposition import PCA

coord = train[['start_latitude','start_longitude']]
pca = PCA(n_components=2)
pca.fit(coord)

coord_pca = pca.transform(coord)

train['coord_pca1'] = coord_pca[:, 0]
train['coord_pca2'] = coord_pca[:, 1]
'''

'''
sns.scatterplot(x='coord_pca2', y='coord_pca1', hue='target', data=train);
plt.show()
'''

'''
inertia_arr = []
from sklearn.cluster import KMeans
k_range = range(2, 16)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(coord)
 
    # Sum of distances of samples to their closest cluster center
    interia = kmeans.inertia_
    print ("k:",k, " cost:", interia)
    inertia_arr.append(interia)
    
inertia_arr = np.array(inertia_arr)

plt.plot(k_range, inertia_arr)
plt.vlines(5, ymin=inertia_arr.min()*0.9999, ymax=inertia_arr.max()*1.0003, linestyles='--', colors='b')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia');
plt.show()
'''
