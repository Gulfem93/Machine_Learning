
#%%
#Kütüphaneler

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#%%
#Verilerin Okunması

veriler = pd.read_csv("satislar.csv")

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#%%
#Verilerin Ön İşlemesi

##Eğitilmesi (Train Test)
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)

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

#%%
#Görselleştirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)