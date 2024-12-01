import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math


def natural_gas_estimate(KGUP, yesterday_price, last_week_price):
    result = 604.57 * math.log(KGUP) + 0.14 * yesterday_price + 0.22 * last_week_price - 3802.96
    return result


P_cost = 1778       # Örnek olarak 100 alınmıştır
P_estimated = natural_gas_estimate(KGUP, yesterday_price, last_week_price)  # Regresyon modeliyle tahmin edilen fiyat
P_max = 3000       # Maksimum teklif fiyatı
Q_max = 24680        # Maksimum üretim miktarı (Mwh)

mu = P_estimated

# CDF değerleri
p1 = 0.01  # P_cost için
p2 = 0.99  # P_max için

# Z-skorları
z1 = norm.ppf(p1) 
z2 = norm.ppf(p2)

sigma1 = (P_cost - mu) / z1
sigma2 = (P_max - mu) / z2
sigma = (abs(sigma1) + abs(sigma2)) / 2

# Fiyat aralığı
price_range = np.linspace(P_cost, P_max, 1000)

# Normal dağılımın CDF'si
cdf_values = norm.cdf(price_range, loc=mu, scale=sigma)

# Üretim miktarı (CDF değerlerini maksimum üretim miktarı ile çarpıyoruz)
production = Q_max * cdf_values

plt.plot(price_range, production)
plt.title('Teklif Fonksiyonu (Normal Dağılım CDF)')
plt.xlabel('Fiyat')
plt.ylabel('Üretim Miktarı')
plt.grid(True)
plt.show()