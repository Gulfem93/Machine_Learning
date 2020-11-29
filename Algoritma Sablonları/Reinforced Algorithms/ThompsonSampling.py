import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000   # Tıklama Sayısı
d = 10      # 10 ilan sayısı

birler = [0] * d    # Ödül olarak birlerin gelmesi
sifirlar = [0] * d  # Ödül olarak sıfırların gelmesi

secilenler = []
toplam = 0

for n in range(0, N): 
    ad = 0   # Seçilen ilan
    max_th = 0
    
    for i in range(0, d):
        rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i
        
    secilenler.append(ad)
    odul = veriler.values[n, ad]
    
    if odul == 1:
        birler[ad] = birler[ad] + 1
    else:
        sifirlar[ad] = sifirlar[ad] + 1
        
    toplam = toplam + odul
print("Toplam Odul")
print(toplam)

plt.hist(secilenler)
plt.show()
        


























