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
from sklearn.metrics import r2_score #(R-square (R-Kare))

#%%
#Verilerin Okunması

veriler = pd.read_csv("maaslar_yeni.csv")
#Sadece tek bir kolonu aldım
x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

#%%
#Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

model = sm.OLS(lin_reg.predict(X), X)
print("\nLinear Regression\n")
print(model.fit().summary())
#%%
#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# 2. dereceden polinomal regresyon
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

model = sm.OLS(lin_reg2.predict(x_poly), X)
print("\n2. dereceden polinomal regresyon\n")
print(model.fit().summary())

# 4. dereceden polinomal regresyon
poly_reg2 = PolynomialFeatures(degree = 4)
x_poly2 = poly_reg2.fit_transform(X)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly2, Y)

model = sm.OLS(lin_reg4.predict(x_poly2), X)
print("\n4. dereceden polinomal regresyon\n")
print(model.fit().summary())

#%% 
#Support Vector Regression (SVR)
from sklearn.svm import SVR

sc = StandardScaler()
sc2 = StandardScaler()
svr_reg = SVR(kernel = 'rbf')

x_sc = sc.fit_transform(X)
y_sc = np.ravel(sc2.fit_transform(Y))

svr_reg.fit(x_sc, y_sc)

model = sm.OLS(svr_reg.predict(x_sc), x_sc)
print("\nSupport Vector Regression (SVR)\n")
print(model.fit().summary())

#%%
#Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state = 0)
tree_reg.fit(X, Y)

model = sm.OLS(tree_reg.predict(X), X)
print("\nDecision Tree\n")
print(model.fit().summary())

#%%
#Random Forrest Regression

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X, Y.ravel())

model = sm.OLS(rf_reg.predict(X), X)
print("\nRandom Forrest Regression\n")
print(model.fit().summary())

