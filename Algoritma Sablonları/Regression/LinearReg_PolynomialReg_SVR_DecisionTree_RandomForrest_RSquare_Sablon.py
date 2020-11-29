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
#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# 2. dereceden polinomal regresyon
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg2.predict(x_poly), color = 'red')
plt.title('2. dereceden polinomal regresyon')
plt.show()

# 4. dereceden polinomal regresyon
poly_reg2 = PolynomialFeatures(degree = 4)
x_poly2 = poly_reg2.fit_transform(X)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly2, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg4.predict(x_poly2), color = 'red')
plt.title('4. dereceden polinomal regresyon')
plt.show()

#%% 
#Support Vector Regression (SVR)
from sklearn.svm import SVR

sc = StandardScaler()
sc2 = StandardScaler()
svr_reg = SVR(kernel = 'rbf')

x_sc = sc.fit_transform(X)
y_sc = np.ravel(sc2.fit_transform(Y))

svr_reg.fit(x_sc, y_sc)

plt.scatter(x_sc, y_sc, color = 'blue')
plt.plot(x_sc, svr_reg.predict(x_sc), color = 'red')
plt.title('Support Vector Regression (SVR)')
plt.show()

#%%
#Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state = 0)
tree_reg.fit(X, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, tree_reg.predict(X), color = 'red')
plt.title('Decision Tree')
plt.show()

#%%
#Random Forrest Regression

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(X, Y.ravel())

plt.scatter(X, Y, color = 'blue')
plt.plot(X, rf_reg.predict(X), color = 'red')
plt.title('Random Forrest')
plt.show()
 
# R-square (R - kare)
print("\n----------------------\n")
print("Linear Regression")
print(r2_score(Y, lin_reg.predict(X)))

print("\n2. dereceden polinomal regresyon")
print(r2_score(Y, lin_reg2.predict(x_poly)))

print("\n4. dereceden polinomal regresyon")
print(r2_score(Y, lin_reg4.predict(x_poly2)))

print("\nSupport Vector Regression (SVR)")
print(r2_score(y_sc, svr_reg.predict(x_sc)))

print("\nDecision Tree")
print(r2_score(Y, tree_reg.predict(X)))

print("\nRandom Forrest Regression")
print(r2_score(Y, rf_reg.predict(X)))


