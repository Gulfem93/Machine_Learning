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

#Logistic Regression
from sklearn.linear_model import LogisticRegression
#PCA kullanmadan önce
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix

#PCA kullanmadan önce
cm = confusion_matrix(Y_test, Y_pred)
print("PCA kullanmadan önce Logistic Regression")
print(cm)


#%%
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train, Y_train)
X_test_lda = lda.transform(X_test)

# LDA kullandıktan sonra Logistic Regression
log_reg_lda = LogisticRegression(random_state = 0)
log_reg_lda.fit(X_train_lda, Y_train)
predict_lda = log_reg_lda.predict(X_test_lda)

#LDA sonrası / orjinal
cm3 = confusion_matrix(Y_pred, predict_lda)
print("\nLDA sonrası / orjinal Logistic Regression")
print(cm3)








