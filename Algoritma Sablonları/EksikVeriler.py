#%%
#Kütüphaneler

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#%%
#Veri Okuma

veriler = pd.read_csv("eksikveriler.csv")
yas = veriler.iloc[:,3:4].values

#%%
#Veri Ön İşleme

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(yas)
yas = imputer.transform(yas)

#%%
#yada
'''
yas = veriler.iloc[:,1:4].values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(:,1:4)
'''

