import pandas as pd
import numpy as np

# Veriler
veriler = pd.read_csv("Wine.csv")
X = veriler.iloc[:,:13].values
Y = veriler.iloc[:,13].values

# Train test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

# Ölçekleme (Standard Scaler)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
#PCA kullanmadan önce
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)

#PCA kullandıktan sonra
log_reg2 = LogisticRegression(random_state = 0)
log_reg2.fit(X_train2, Y_train)
predict_pca = log_reg2.predict(X_test2)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
#PCA kullanmadan önce
cm = confusion_matrix(Y_test, Y_pred)
print("PCA kullanmadan önce Logistic Regression")
print(cm)

#PCA kullandıktan sonra
cm = confusion_matrix(Y_test, predict_pca)
print("\nPCA kullandıktan sonra Logistic Regression")
print(cm)