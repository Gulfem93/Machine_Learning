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

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#%%
#Verilerin Okunması

veriler = pd.read_csv("eksikveriler.csv")

ulkeler = veriler.iloc[:,0:1].values
boy_kilo_yas = veriler.iloc[:,1:4].values
cinsiyet = veriler[['cinsiyet']]
boy = veriler.iloc[:,1:2]

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
sonuc = pd.concat([ulkeler, boy_kilo_yas], axis = 1)
sonuc = pd.concat([sonuc, cinsiyet], axis = 1)

ulke_boy_kilo_yas = pd.concat([ulkeler, boy_kilo_yas], axis = 1)
ulke_kilo_yas_cinsiyet = pd.concat([sonuc.iloc[:,0:3], sonuc.iloc[:,4:]], axis = 1)

##Eğitilmesi (Train Test)
x_train, x_test, y_train, y_test = train_test_split(ulke_boy_kilo_yas, cinsiyet, test_size = 0.33, random_state = 0)

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
lr = LinearRegression()
lr.fit(x_train, y_train)

##tahmin
tahmin = lr.predict(x_test)



##Boyun bulunması
lr2 = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(ulke_kilo_yas_cinsiyet, boy, test_size = 0.33, random_state = 0)

lr2.fit(x_train, y_train)
tahmin2 = lr2.predict(x_test)




#%%
#Görselleştirme
x_train = x_train.sort_index()      #İndexleri random bir şekilde olan dataframe sıralı hhale dönüştürür
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)
