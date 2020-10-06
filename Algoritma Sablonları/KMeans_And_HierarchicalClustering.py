#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#Veriler
veriler = pd.read_csv('musteriler.csv')

x = veriler.iloc[:,3:].values #bağımsız değişkenler

#%% 
# Kmeans
from sklearn.cluster import KMeans
'''
kmeans = KMeans(n_clusters = 4, init = 'k-means++')

kmeans.fit(x)
print(kmeans.cluster_centers_)
'''

'''
#Kmeans

sonuc = []

for i in range(1,10):
    km = KMeans(n_clusters=i, init = 'k-means++', random_state=123)
    km.fit(x)
    sonuc.append(km.inertia_)

plt.plot(range(1,10), sonuc)
'''

km = KMeans(n_clusters=4, init = 'k-means++', random_state=123)
kmeans_predict = km.fit_predict(x)

kmeans = plt.figure(figsize=(5, 10))
plt.scatter(x[kmeans_predict == 0, 0], x[kmeans_predict == 0, 1], s = 50, c = 'red')
plt.scatter(x[kmeans_predict == 1, 0], x[kmeans_predict == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[kmeans_predict == 2, 0], x[kmeans_predict == 2, 1], s = 50, c = 'blue')
plt.scatter(x[kmeans_predict == 3, 0], x[kmeans_predict == 3, 1], s = 50, c = 'green')
plt.title('Kmeans')
plt.show()

#%% 
# Hierarchical Clustering 

from sklearn.cluster import AgglomerativeClustering
AggloCluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

agglo_predict = AggloCluster.fit_predict(x)
HClustering  = plt.figure(figsize=(5, 10))
#Figure
plt.scatter(x[agglo_predict == 0, 0], x[agglo_predict == 0, 1], s = 50, c = 'red')
plt.scatter(x[agglo_predict == 1, 0], x[agglo_predict == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[agglo_predict == 2, 0], x[agglo_predict == 2, 1], s = 50, c = 'blue')
plt.scatter(x[agglo_predict == 3, 0], x[agglo_predict == 3, 1], s = 50, c = 'green')
plt.title('Hierarchical Clustering ')
plt.show()

#
import scipy.cluster.hierarchy as sch
hdendogrom  = plt.figure(figsize=(10, 10))
dendogrom = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.show()





















