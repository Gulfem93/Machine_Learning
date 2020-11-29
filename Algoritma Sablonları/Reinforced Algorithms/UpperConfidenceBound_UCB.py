import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#Veriler 
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#%%
#Upper Confidence Bound (UCB)

N = 10000
d = 10

#Ri(n)
oduller = [0]*d     # ilk başta bütün ilanların ödülü 0

#Ni(n)
tiklananlar = [0]*d     # O ana kadarki tıklamalar

toplam = 0  # Toplam ödül
secilenler = []

for n in range(1, N):
    
    max_ucb = 0
    ad = 0
    
    for i in range(0, d):   
        # max_ucb sahip ilanları buluruz
        # bütün ilanların tek tek ihtimallerine bakarız
        # 10 ilana teker teker bak. En fazla ucb bulur
        
        if (tiklananlar[i] > 0):
            # Ilk başta tıklama kontrolu
            
            ortalama = oduller[i]/tiklananlar[i]    # Ortalama = Ri(n)/Ni(n)
            delta = math.sqrt(3/2 *math.log(n)/tiklananlar[i])   # Aşagı yukarı oynama potansiyeli --> di(n) = math.sqrt(3/2 *log(n)/Ni(n))
            ucb = delta + ortalama
            
        else:
            ucb = N*10000
        
        if max_ucb < ucb:
            # En yüksek ucb ilan benim tıklayacağım ilan
            # max_ucb benim secilen ilananım olacak
            
            max_ucb = ucb
            ad = i
            
    secilenler.append(ad)
    tiklananlar[ad] = tiklananlar[ad] + 1   # Hangi ilana tıkladıysam 1 arttırmalıyım
    odul = veriler.values[n, ad]    # Tıkladığım yerdeki ödül
    oduller[ad] = oduller[ad] + odul     # Odulleri odul kadar attırır
    toplam = toplam + odul  


print('Toplam Ödül')
print(toplam)

plt.hist(secilenler)
plt.show()














