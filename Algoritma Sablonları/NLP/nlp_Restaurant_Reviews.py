# Kütüphaneler
import numpy as np
import pandas as pd
import re
import nltk

yorumlar = pd.read_csv("Restaurant_Reviews.csv", error_bad_lines=False)

#Stopwords -> Anlamı olmayan kelimelerdir ör ingilizcede : the, a, an, ....
nltk.download('stopwords')
from nltk.corpus import stopwords

#Stemmer -> Kelimeyi köklerine ayirir
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#Preprocessing(Ön İşleme)
derlem = []
for i in range(716):
    
    # İmla işaretlerini ve noktalama işareterini kaldırma
    yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
    
    #Küçük harfe çevrildi
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    
    yorum = ' '.join(yorum)
    
    derlem.append(yorum)


#Feature Extraction(Öznitelik Çıkarımı)
#Bag of Word (BOW)
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features = 1000)

x = CV.fit_transform(derlem).toarray()  #Bağımsız değişken
y = yorumlar.iloc[:,1].values           #Bağımlı değişken

#Machine Learning
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

##Clasification da Naive Bayes uygulandı
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

predict = gnb.predict(x_test)

##Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict)
print(cm)


























