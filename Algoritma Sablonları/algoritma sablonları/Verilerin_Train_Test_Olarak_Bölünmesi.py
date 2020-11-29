# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 19:05:53 2020

@author: IŞIK
"""

#%%
#Kütüphaneler

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#%%
#Verilerin Okunması

veriler = pd.read_csv("eksikveriler.csv")

ulkeler = veriler.iloc[:,0:1].values
boy_kilo_yas = veriler.iloc[:,1:4].values
cinsiyet = veriler.iloc[:,4].values

#%%
#Verilerin Ön İşlemesi

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
boy_kilo_yas = imputer.fit_transform(boy_kilo_yas)

##Kategorize Edildi
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

ulkeler = ohe.fit_transform(ulkeler).toarray()
cinsiyet = le.fit_transform(cinsiyet)

##Dataframe Dönüştürüldü
ulkeler = pd.DataFrame(data = ulkeler, index = range(22), columns = ["fr", "tr", "us"])
cinsiyet = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])
boy_kilo_yas = pd.DataFrame(data = boy_kilo_yas, index = range(22), columns = ["boy", "kilo", "yas"])

##DataFrameler birleştirildi
sonuc = pd.concat([ulkeler, boy_kilo_yas], axis = 1)
sonuc = pd.concat([sonuc, cinsiyet], axis = 1)

ulke_boy_kilo_yas = pd.concat([ulkeler, boy_kilo_yas], axis = 1)

##Eğitilmesi (Train Test)
x_train, x_test, y_train, y_test = train_test_split(ulke_boy_kilo_yas, cinsiyet, test_size = 0.33, random_state = 0)
