#%%
#Kütüphaneler

import pandas as pd

#%%
#Veri Okuma

veriler = pd.read_csv("eksikveriler.csv")

yas = veriler[["yas"]]
boy = veriler.iloc[:,1:2]

