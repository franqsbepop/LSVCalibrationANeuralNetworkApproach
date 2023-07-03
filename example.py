import numpy as np
from scipy.stats import norm
from scipy.special import ndtri

def minmaxvar(l, u):
    result = 1 - (1 - u**(1/(l+1)))**(1+l)
    return result

def Wang(l, u):
    result = norm.cdf(ndtri(u)+l)
    return result

def comp_wang_minmaxvar(l, u):
    return Wang(l, minmaxvar(l, u))

def comp_minmaxvar_wang(l, u):
    return minmaxvar(l, Wang(l, u))

for l in np.arange(0, 1, 0.1):
    for u in  np.arange(0, 1, 0.1):
        y = comp_wang_minmaxvar(l, u) - comp_minmaxvar_wang(l, u)
        print(y)

        