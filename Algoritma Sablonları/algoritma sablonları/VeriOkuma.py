# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:27:33 2020

@author: IŞIK
"""

#%%
#Kütüphaneler

import pandas as pd

#%%
#Veri Okuma

veriler = pd.read_csv("eksikveriler.csv")

yas = veriler[["yas"]]
boy = veriler.iloc[:,1:2]

