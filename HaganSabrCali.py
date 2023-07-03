from Hagan2002LognormalSABR_ import Hagan2002LognormalSABR
import numpy as np
from scipy.stats import norm
import os
#From get data file
from get_data import get_maturities
from get_data import get_strikes
from get_data import get_spot
from get_data import get_bids
from get_data import get_asks

#Name of Data file
real_data = "7Maturities20Kv.2"
# get the parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))

#Set finsurf_iV dummy variable
finsurf_iV = [np.array([0.45942663, 0.43595129, 0.4106667 , 0.3916855 , 0.36725782,
       0.34877793, 0.32722994, 0.30750754, 0.28821458, 0.26948812,
       0.24972536, 0.22953326, 0.2082357 , 0.18591902, 0.1647328 ,
       0.14894247, 0.14243214, 0.14260997, 0.14524687, 0.14969965]), np.array([0.37203963, 0.35842147, 0.34444481, 0.33226158, 0.31995406,
       0.30664746, 0.29045663, 0.27961627, 0.26596093, 0.25225101,
       0.23632418, 0.22315411, 0.20802311, 0.19250624, 0.17745269,
       0.16410129, 0.15354445, 0.1475732 , 0.14518668, 0.14501527]), np.array([0.34111019, 0.33029136, 0.31936234, 0.31191202, 0.29981583,
       0.28980064, 0.27852654, 0.26810303, 0.25621674, 0.24452395,
       0.2327292 , 0.22056294, 0.20737693, 0.19539918, 0.18244364,
       0.17160698, 0.16111792, 0.15389765, 0.14867029, 0.14617052]), np.array([0.32115245, 0.31250497, 0.30380628, 0.29752104, 0.28831981,
       0.27895831, 0.26937841, 0.25950787, 0.25188693, 0.24067335,
       0.22986102, 0.21868861, 0.20730407, 0.19876685, 0.18749151,
       0.17669879, 0.16706311, 0.15876038, 0.15259102, 0.14944133]), np.array([0.29988688, 0.29394852, 0.28709687, 0.28174303, 0.27438831,
       0.2668199 , 0.25897052, 0.25090933, 0.24461823, 0.23582841,
       0.22742189, 0.21786009, 0.20862894, 0.20165775, 0.19353962,
       0.1828313 , 0.17416617, 0.1663937 , 0.15986327, 0.15524378]), np.array([0.28917704, 0.2838119 , 0.27755028, 0.2764531 , 0.26676982,
       0.26017067, 0.2532921 , 0.24612184, 0.24059501, 0.23260395,
       0.22475864, 0.21675324, 0.2085936 , 0.20245361, 0.19404068,
       0.18578755, 0.17780828, 0.17036794, 0.1639598 , 0.15889945])]

def get_alpha_rho_nu(real_data, finsurf_iV):
       #Get parameters  from the data
       mat = get_maturities(real_data)
       K = get_strikes(real_data)
       S0 = get_spot(real_data)
       v_sln = finsurf_iV
       #Do the Sabr Fit
       sabr = Hagan2002LognormalSABR(f=S0, shift=0, t=mat, beta=1)
       [alpha, rho, volvol] = sabr.fit(K, v_sln)
       return alpha, rho, volvol

#Print rho, alpha, volvol
[alpha, rho, volvol] = get_alpha_rho_nu(real_data, finsurf_iV)
##Print
print(alpha)
print(rho)
print(volvol)
#  returns [0.025299981543599154, -0.24629917636394097, 0.2908005625794777]

#Define necessary functions
def _x(rho, z):
    """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
    a = (1 - 2*rho*z + z**2)**.5 + z - rho
    b = 1 - rho
    return np.log(a / b)

#Lognormal Vol
def lognormal_vol(strikes, f, t, alpha, beta, rho, volvol):
    """
    Hagan's 2002 SABR lognormal vol expansion.
    The strike k can be a scalar or an array, the function will return an array
    of lognormal vols.
    """
    vols = []
    for  k in strikes:
       eps = 1e-07
       logfk = np.log(f / k)
       fkbeta = (f*k)**(1 - beta)
       a = (1 - beta)**2 * alpha**2 / (24 * fkbeta)
       b = 0.25 * rho * beta * volvol * alpha / fkbeta**0.5
       c = (2 - 3*rho**2) * volvol**2 / 24
       d = fkbeta**0.5
       v = (1 - beta)**2 * logfk**2 / 24
       w = (1 - beta)**4 * logfk**4 / 1920
       z = volvol * fkbeta**0.5 * logfk / alpha
       # if |z| > eps
       if abs(z) > eps:
              vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
              vz = vz*100
              vols.append(vz)
       # if |z| <= eps
       else:
              v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
              v0 = v0*100
              vols.append(v0)
              # print('THIS is v0', v0)
    return vols

# ln_vols = lognormal_vol(K[0], f=4567, t = 0.17,alpha = alpha, beta=0.5, rho=rho, volvol=volvol)
# print('ln_vols', ln_vols)

#LogNormal Cali
def lognormal_call(strikes, f, t, v, r, cp='call'):
    """Compute an option premium using a lognormal vol."""
    prices = []
    for i, k in enumerate(strikes):
       # print(v[i])
       # print(k)
       if k <= 0 or f <= 0 or t <= 0 or v[i] <= 0:
              return 0.
       d1 = (np.log(f/k) + v[i]**2 * t/2) / (v[i] * t**0.5)
       d2 = d1 - v[i] * t**0.5
       if cp == 'call':
              pv = np.exp(-r*t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
       elif cp == 'put':
              pv = np.exp(-r*t) * (-f * norm.cdf(-d1) + k * norm.cdf(-d2))
       else:
              pv = 0
       prices.append(pv)
    return prices

def get_prices_sabr_cali(real_data, finsurf_iV):
       #Set parameters
       K = get_strikes(real_data)
       mat = get_maturities(real_data)
       S0 = get_spot(real_data)
       #Get rho, alpha, volvol
       [alpha, rho, volvol] = get_alpha_rho_nu(real_data, finsurf_iV)
       #Price surface
       price_surface  = []
       for i in range(len(K)):
              ln_vols = lognormal_vol(K[i], f=S0, t = mat[i], alpha = alpha, beta=1, rho=rho, volvol=volvol)
              # print(mat[i])
              prices_mat = np.array(lognormal_call(K[i], f=S0, t=mat[i], v = ln_vols, r=0, cp='call'))
              price_surface.append(prices_mat)

       return alpha, rho, volvol, price_surface

print('GET PRICE', get_prices_sabr_cali(real_data, finsurf_iV)[-1])



