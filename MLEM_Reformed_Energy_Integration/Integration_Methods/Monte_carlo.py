import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special as sp
import math

MeCsq = 0.5109989461
sig = 0.1

def gaussian(mean, sig, val):
    return 1.0/(sig * np.sqrt(2 * np.pi)) * np.exp( - (val - mean)**2 / (2 * sig**2) )

def generate_e(E_m, sig):
    E = 0
    while ( E <= 0):
        E = np.random.normal( E_m , sig )
    return E

def scattering_angle(E1, E2):
    s = 1 + MeCsq * ( 1.0/(E1+E2) - 1.0/(E2) )

    while ( abs(s) >= 1 ):
        E1 = generate_e(E1_mean, sig)
        E2 = generate_e(E2_mean, sig)

        s = 1 + MeCsq * ( 1.0/(E1+E2) - 1.0/(E2) )

    return np.arccos(s)

def uncertainty_double_scatter(E1, E2, ue ):
    return np.sqrt( abs( ( ((E1**4) + 4 * (E1**3) * E2 + 4 * (E1**2) * (E2**2) + (E2**4)) * MeCsq * (ue**2)) / ( E1 * (E2**2) * (E1+E2)**2 * ( 2 * E2 * (E1+E2) - E1 * MeCsq ) ) ) )

N = 50000

E1_mean = 0.3366
E2_mean = 0.3274

theta_mean = scattering_angle(E1_mean, E2_mean)
theta_sig = uncertainty_double_scatter(E1_mean, E2_mean, sig)

E1_s = [ generate_e(E1_mean, sig) for k in range(N) ]
E2_s = [ generate_e(E2_mean, sig) for k in range(N) ]

thetas = [ scattering_angle(E1, E2) for E1, E2 in zip(E1_s, E2_s) ]

count, bins, ignored = plt.hist(thetas, N/50 , normed=True)

x_space = np.linspace(0, np.pi, 100)

plt.plot(bins, gaussian(theta_mean, theta_sig, bins) , linewidth=2, color='r')
plt.show()
