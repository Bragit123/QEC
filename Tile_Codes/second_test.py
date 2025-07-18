import numpy as np
from scipy.special import erf, erfinv

th = np.sqrt(np.pi) / 2.0
def p_corr(sigma):
    eps = 1e-10
    return erf(th/(np.sqrt(2)*sigma+eps))

sigmas = np.array([0.0, 0.5, 0.8, 1.0, 2.0, 10.0])
print(p_corr(sigmas))