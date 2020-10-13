# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 00:34:41 2020

@author: IŞIK
"""

#%%
#Kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#%%
#Verilerin Okunması

veriler = pd.read_csv("eksikveriler.csv")

ulkeler = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,4:]
boy = veriler.iloc[:,1:2]
veri = veriler.iloc[:,1:4].values
boy_kilo_yas = veriler.iloc[:,1:4]

#%%
#Verilerin Ön İşlemesi

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
boy_kilo_yas = imputer.fit_transform(boy_kilo_yas)

##Kategorize Edildi
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

ulkeler = ohe.fit_transform(ulkeler).toarray()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

##Dataframe Dönüştürüldü
ulkeler = pd.DataFrame(data = ulkeler, index = range(22), columns = ["fr", "tr", "us"])
cinsiyet = pd.DataFrame(data = cinsiyet[:,0:1], index = range(22), columns = ["cinsiyet"])
boy_kilo_yas = pd.DataFrame(data = boy_kilo_yas, index = range(22), columns = ["boy", "kilo", "yas"])

##DataFrameler birleştirildi
veri = pd.concat([ulkeler, boy_kilo_yas.iloc[:,1:]], axis = 1)
veri = pd.concat([veri, cinsiyet], axis = 1)

##Eğitilmesi (Train Test)
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size = 0.33, random_state = 0)

##StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test =sc.fit_transform(x_test)
'''
##StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#%%
#Tamin yaparım

lr = LinearRegression()
lr.fit(x_train, y_train)

##tahmin
tahmin = lr.predict(x_test)


#%%
#Bacward Elimination

X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)

model = sm.OLS(boy, X_l).fit()
print(model.summary())




