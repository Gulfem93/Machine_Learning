# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#%%
#Veriler
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

#%%
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%% 
#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

knn_predict = knn.predict(X_test)

cm = confusion_matrix(y_test, knn_predict)
print(cm)

#%%
#SVC
from sklearn.svm import SVC

svc_rbf = SVC(kernel='rbf')
svc_linear = SVC(kernel = "linear")

svc_rbf.fit(X_train, y_train)
svc_linear.fit(X_train, y_train)

svc_predict_rbf = svc_rbf.predict(X_test)
svc_predict_linear = svc_linear.predict(X_test)

cm_rbf = confusion_matrix(y_test, svc_predict_rbf)
print("SVC - rbf")
print(cm_rbf)

cm_linear = confusion_matrix(y_test, svc_predict_linear)
print('SVC - Linear')
print(cm_linear)





































