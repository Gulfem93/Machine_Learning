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
