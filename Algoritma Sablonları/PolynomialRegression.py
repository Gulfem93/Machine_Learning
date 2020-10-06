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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#%%
#Verilerin Okunması

veriler = pd.read_csv("maaslar.csv")
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#%%
#Veri Hazırlık

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg.predict(X), color = 'red')
plt.title('degree = 1 (linear regression)')
plt.show()

#%%
#Polynomial Regression

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg2.predict(x_poly), color = 'red')
plt.title('degree = 2')
plt.show()

#degree(derece = 4 iken)

poly_reg2 = PolynomialFeatures(degree = 4)
x_poly2 = poly_reg2.fit_transform(X)

lin_reg4 = LinearRegression()
lin_reg4.fit(x_poly2, Y)

plt.scatter(X, Y, color = 'blue')
plt.plot(X, lin_reg4.predict(x_poly2), color = 'red')
plt.title('degree = 4')
plt.show()
