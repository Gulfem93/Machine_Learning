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
kmeans = KMeans(n_clusters = 4, init = 'k-means++')

kmeans.fit(x)
print(kmeans.cluster_centers_)

#Kmeans

sonuc = []

for i in range(1,10):
    km = KMeans(n_clusters=i, init = 'k-means++', random_state=123)
    km.fit(x)
    sonuc.append(km.inertia_)

plt.plot(range(1,10), sonuc)

#%% 
#Figure

kmeans_predict = kmeans.predict(x)

figure = plt.figure(figsize = (10, 10))
plt.scatter(x[:,0], x[:,1], c = kmeans_predict, s = 50, cmap = 'viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)


























