# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
#Veriler
veriler = pd.read_csv('musteriler.csv')

x = veriler.iloc[:,3:].values #bağımsız değişkenler
#%% 
# Hierarchical Clustering 

from sklearn.cluster import AgglomerativeClustering
AggloCluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

agglo_predict = AggloCluster.fit_predict(x)


#%% 
#Figure

plt.scatter(x[agglo_predict == 0, 0], x[agglo_predict == 0, 1], s = 50, c = 'red')
plt.scatter(x[agglo_predict == 1, 0], x[agglo_predict == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[agglo_predict == 2, 0], x[agglo_predict == 2, 1], s = 50, c = 'blue')
plt.scatter(x[agglo_predict == 3, 0], x[agglo_predict == 3, 1], s = 50, c = 'green')
plt.show()

#
import scipy.cluster.hierarchy as sch
figure = plt.figure(figsize=(10,20))
dendogrom = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.show()





















