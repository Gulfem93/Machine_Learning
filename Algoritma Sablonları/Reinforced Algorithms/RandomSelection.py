import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
#%%
#Veriler
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#%%
#Random Selection

## Rastgele secimler yapılır yapılan seçimler eğer orjinal veriler ile eş ise ödül kazanırız. 
## Kazandığımız ödülleri toplayarak ilerleriz.

secilenler = []
toplam = 0
N = 10000
d = 10

for n in range(0, N):
    ad = random.randrange(d)    # Tıkladığımız ilandan 10 tane random değer üret
    secilenler.append(ad)
    odul = veriler.values[n, ad]    # Verilerdeki n. satır 1 ise odul = 1
    toplam = toplam + odul

plt.hist(secilenler)
plt.title('Rastgele secilim')
plt.show()























