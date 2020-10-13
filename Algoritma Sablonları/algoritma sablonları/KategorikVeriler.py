# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:49:33 2020

@author: IŞIK
"""

#%%
#Kütüphaneler

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#%%
#Veri Okuma

veriler = pd.read_csv("eksikveriler.csv")

boy_kilo_yas = veriler.iloc[:,1:4].values
ulkeler = veriler.iloc[:,0:1].values
cinsiyet = veriler.iloc[:,-1].values

#%%
#Veri Ön İşleme

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
boy_kilo_yas = imputer.fit_transform(boy_kilo_yas)

##Kategorize edildi
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

ulke = le.fit_transform(ulkeler)
ulke1 = ohe.fit_transform(ulkeler).toarray()

cinsiyet = le.fit_transform(cinsiyet)

# DataFrame yapma
boy_kilo_yas = pd.DataFrame(data = boy_kilo_yas, index = range(22), columns = ["boy", "kilo", "yas"])
cinsiyet = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])
ulkeler = pd.DataFrame(data = ulke1, index = range(22), columns = ["fr", "tr", "us"])

sonuc = pd.concat([ulkeler, boy_kilo_yas], axis = 1)
sonuc = pd.concat([sonuc, cinsiyet], axis = 1)