# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 01:04:32 2020

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


#%%
#Verilerin Okunması

veriler = pd.read_csv("maaslar.csv")
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#%%
#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('linear regression')
plt.show()

#%%
#Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state = 0)
Z = X + 0.5
K = X - 0.4
tree_reg.fit(X, Y)

print(tree_reg.predict([[6.6]]))
print(tree_reg.predict([[6.3]]))
print(tree_reg.predict([[6.8]]))

plt.scatter(X, Y, color = 'blue')
plt.plot(X, tree_reg.predict(X), color = 'red')
plt.plot(X, tree_reg.predict(K), color = 'yellow')
plt.plot(X, tree_reg.predict(Z), color = 'green')
plt.show()

#%%
# Random Forrest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_reg.fit(X, Y)

print(rf_reg.predict([[6.6]]))
print(rf_reg.predict([[6.3]]))
print(rf_reg.predict([[6.8]]))

plt.scatter(X, Y, color = 'blue')
plt.plot(X, rf_reg.predict(X), 'red')
plt.show()