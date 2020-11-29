import pandas as pd
import numpy as np

veriler = pd.read_csv("Churn_Modelling.csv")

X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13:].values

#LabelEncoder and OneHotEncoder

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float), [1])], remainder="passthrough")
X = ohe.fit_transform(X)

#train test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#%%
# Yapay Sinir Ağı
import keras 
from keras.models import Sequential
from keras.layers import Dense

deep_learning = Sequential()
deep_learning.add(Dense(6, init="uniform", activation = "relu", input_dim = 12))
deep_learning.add(Dense(6, init="uniform", activation = "relu"))
deep_learning.add(Dense(1, init="uniform", activation = "sigmoid"))

deep_learning.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
deep_learning.fit(x_train, y_train, epochs = 50)             
         
y_predict = deep_learning.predict(x_test)     

#%%
#SVM 
from sklearn.svm import SVC
svc = SVC(kernel = "rbf")
svc.fit(x_train, y_train)
svc_predict = svc.predict(x_test)

#%%
# XGBoost
from xgboost import XGBClassifier
xg_classifier = XGBClassifier()

xg_classifier.fit(x_train, y_train)
xg_predict = xg_classifier.predict(x_test)

#Başarı ölçümü
from sklearn.model_selection import cross_val_score
xg_basari = cross_val_score(estimator = xg_classifier, X = x_train, y = y_train, scoring = "accuracy", cv = 4)
svc_basari = cross_val_score(estimator = svc, X = x_train, y = y_train, scoring = "accuracy", cv = 4)
print("---------------------")
print("XGBoost Başarı:")
print(xg_basari.mean())
print("---------------------")
print("SVC Başarı:")
print(svc_basari.mean())
