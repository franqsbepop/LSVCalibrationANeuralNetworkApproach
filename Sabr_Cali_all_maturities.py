import numpy as np
from scipy.stats import norm
import os
from pysabr import Hagan2002LognormalSABR
from pysabr import hagan_2002_lognormal_sabr as hagan2002
#From get data file
from get_data import get_maturities
from get_data import get_strikes
from get_data import get_spot
from get_data import get_bids
from get_data import get_asks
from get_data import get_mid_prices
from distutils.log import error
import pandas as pd
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from py_vollib.black_scholes.implied_volatility import implied_volatility as implVola
from py_vollib.black_scholes_merton import black_scholes_merton
from scipy.optimize import minimize_scalar

#Name of Data file
real_data = "Cali_6mat20k"
#Name of Cali file
calibration_name = 'SP500_calibeta1' 
# get the parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))
#File path for finsurf instance
finsurf_path = parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\histData.npz'

#READ AND PRINT NPZ FILE
data = np.load(finsurf_path)

#DATA FOR GRAPHS BELOW
#PRICES_DATA
# prices_data = data['p_data']
# prices_data = get_mid_prices(real_data)
# #IV_DATA
finsurf_iV = data['iV_data']
Mat = get_maturities(real_data)
S0 = get_spot(real_data)
K = get_strikes(real_data)


def get_alpha_rho_nu(real_data, finsurf_iV):    
       #Get parameters  from the data
       alpharhovolvol = []
       mat = get_maturities(real_data)
       K = get_strikes(real_data)
       S0 = get_spot(real_data)
       v_sln = finsurf_iV
       v_sln1 = v_sln
       #Do the Sabr Fit
       for i in range(len(mat)):
            sabr = Hagan2002LognormalSABR(f=S0, shift=0, t=mat[i], beta=1)
            x100logvol =  v_sln1[i]*100
            temp = sabr.fit(K[i], x100logvol)
            alpharhovolvol.append(temp)
       return alpharhovolvol

#Print rho, alpha, volvol
alpharhovolvol = get_alpha_rho_nu(real_data, finsurf_iV)

#Print
# print('ALPHA, RHO, VOLVOL', alpharhovolvol)


#Define necessary functions
def _x(rho, z):
    """Return function x used in Hagan's 2002 SABR lognormal vol expansion."""
    a = (1 - 2*rho*z + z**2)**.5 + z - rho
    b = 1 - rho
    return np.log(a / b)

#Lognormal Vol
def lognormal_call(strikes, f, t, v, r, cp='call'):
    """Compute an option premium using a lognormal vol."""
    prices = []
    for i, k in enumerate(strikes):
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
       alpharhovolvol = get_alpha_rho_nu(real_data, finsurf_iV)
       #Price surface
       price_surface  = []
       vola_surface = []
       py_volib_surface = []
       for i in range(len(mat)):
              [alpha, rho, volvol] = alpharhovolvol[i]
              ln_vols = [hagan2002.lognormal_vol(k, f=S0, t = mat[i], alpha = alpha, beta=1, rho=rho, volvol=volvol) for k in K[i]]
              prices_mat = np.array(lognormal_call(K[i], f=S0, t=mat[i], v = ln_vols, r=0, cp='call'))
              #Calculate implied vols with py_volib package
              py_volib_vols = [implVola(prices_mat[x], S0, K[i][x], mat[i], 0.0, 'c') for x in range(len(K[i]))]
              #Append vola surface and price surface
              vola_surface.append(ln_vols)
              price_surface.append(prices_mat)
              py_volib_surface.append(py_volib_vols)
       return price_surface, vola_surface, py_volib_surface

# print(finsurf_iV, get_prices_sabr_cali(real_data, finsurf_iV)[1], get_prices_sabr_cali(real_data, finsurf_iV)[2])=



#Define basic properties
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
mat = get_maturities(real_data)


#Get Implied Vol of SABR calibrated Model
vol_sabr_model = get_prices_sabr_cali(real_data, finsurf_iV)[1]

#Define getting bid and ask prices for volatility surface
def bid_ask_call(vol, strike, S0, T,  gamma):    
    increment = gamma * vol / np.sqrt(strike)
    bid_price = black_scholes_merton('c', S=S0, K=strike, t=T, r=.0, sigma=vol, q=increment)
    ask_price = black_scholes_merton('c', S=S0, K=strike, t=T, r=.0, sigma=vol, q=-increment)
    return bid_price, ask_price

#define sum of squred error 
def sum_of_squares(gamma, vol, strike, S0, T, real_bid, real_ask): 
       bid_price, ask_price = bid_ask_call(vol, strike, S0, T,  gamma)
       return (real_bid - bid_price)**2 + (real_ask - ask_price)**2


def bid_ask_cali(real_data, vol_sabr_model):
       K = get_strikes(real_data)
       Wang_impliedliquiditysurface_MK = []
       Wang_bidsmodel_MK = []
       sensitivity_Wang_bidsmodel_MK = []
       Wang_asksmodel_MK = []
       sensitivity_Wang_asksmodel_MK = []
       for i in range(len(data['mc_prices_model'])):
              mat = get_maturities(real_data)[i]
              #Create object to store Wang results
              Wang_impliedliquiditysurface_K = []
              Wang_bidsmodel_K = []
              sensitivity_Wang_bidsmodel_K = []
              Wang_asksmodel_K = []
              sensitivity_Wang_asksmodel_K = []
              for k in range(len(K[0])):
                     #Get the MC payoff and the real bid and ask
                     spot = get_spot(real_data)
                     strike = K[i][k]
                     vol_sabr_MK = vol_sabr_model[i][k]
                     real_bid = get_bids(real_data)[i][k]
                     real_ask = get_asks(real_data)[i][k]
                     print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
                     result_wang =  minimize_scalar(lambda l: sum_of_squares(gamma = l, vol =  vol_sabr_MK,  strike =  strike, S0 = spot, T= mat, real_bid = real_bid, real_ask = real_ask), bounds=(0,1), method = 'bounded')
                     gamma_wang_min = result_wang.x 
                     #get the bid-ask prices for this value of gamma and calculate sensitivities w.r.t these parameters
                     cali_bid_price, cali_ask_price = bid_ask_call( vol =  vol_sabr_MK,   strike =  strike, S0 = spot, T = mat,  gamma = gamma_wang_min)
                     #sensitivity
                     h = 0.0001
                     # 
                     cali_bid_priceh , cali_ask_priceh = bid_ask_call( vol =  vol_sabr_MK,   strike =  strike, S0 = spot, T = mat,  gamma = (gamma_wang_min+h))
                     cali_bid_price__h , cali_ask_price__h = bid_ask_call( vol =  vol_sabr_MK,   strike =  strike, S0 = spot, T = mat,  gamma = (gamma_wang_min-h))
                     sensitivity_lambda_bidwang =  (cali_bid_priceh - cali_bid_price__h)/2*h
                     sensitivity_lambda_askwang =  (cali_ask_priceh - cali_ask_price__h)/2*h     
                     #Append all these values into designated list
                     Wang_impliedliquiditysurface_K.append(gamma_wang_min)
                     Wang_bidsmodel_K.append(cali_bid_price)
                     sensitivity_Wang_bidsmodel_K.append(sensitivity_lambda_bidwang)
                     Wang_asksmodel_K.append(cali_ask_price)
                     sensitivity_Wang_asksmodel_K.append(sensitivity_lambda_askwang)
                     ####
              Wang_impliedliquiditysurface_MK.append(Wang_impliedliquiditysurface_K)
              Wang_bidsmodel_MK.append(Wang_bidsmodel_K)
              sensitivity_Wang_bidsmodel_MK.append(sensitivity_Wang_bidsmodel_K)
              Wang_asksmodel_MK.append(Wang_asksmodel_K)
              sensitivity_Wang_asksmodel_MK.append(sensitivity_Wang_asksmodel_K)
       return Wang_bidsmodel_MK, Wang_asksmodel_MK, Wang_impliedliquiditysurface_MK, sensitivity_Wang_bidsmodel_MK, sensitivity_Wang_asksmodel_MK


       
#Lets Plot the results

#Plot implied volatility model vs data
for i in range(len(mat)):
    x_points = moneyness[i]
    y1_points = finsurf_iV[i]
    y2_points = vol_sabr_model[i]
    error_iv =  y1_points - y2_points

#     print('The error average absolute error for maturity {}'.format(mat[i]), np.mean(np.abs(error_iv)))
#     print('MAX ERROR', np.max(np.abs(error_iv)))
#     print('INDEX MAX ERROR', list(np.abs(error_iv)).index(np.max(np.abs(error_iv))))
#     print('Squared error', (error_iv*10000)**2)
    #Figure of Smiles
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, y1_points, label = 'Data implied vol')
    plt.plot(x_points, y2_points, label = 'SABR fit implied vol')
    plt.ylabel('Implied Volatility')
    plt.xlabel('Moneyness')
    plt.title('Volatility smile (T = {})'.format(mat[i]))
    plt.legend()
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\SABR_implied_vol_smile_mat{}.eps".format(mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.show()
    plt.close()
    
    #Figure of Error
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, error_iv)
    plt.ylabel('Error')
    plt.xlabel('Moneyness')
    plt.title('Implied volatility error (T = {})'.format(mat[i]))
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\SABR_implied_volatility_error_mat{}.eps".format(mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.show()
    plt.close()

#Plot the implied volatility squared error after Calibration
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
# defining surface and axes
y = np.array(get_maturities(real_data))
x = moneyness[0]
x, y = np.meshgrid(x, y)
z1 = finsurf_iV
z2 = vol_sabr_model
z = (z1 - z2)**2


fig = plt.figure(figsize=(7, 5))
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
# syntax for plotting
ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='yellow')
ax.set_title('Implied volatility squared error')
ax.set_ylabel('Time to maturity (years)')
ax.set_xlabel('Moneyness')
name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_volatility_squared_error_surface_SABR.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.show()
plt.close()


#Plot Bid and Ask IV
#run the bid-askj Calibration
Wang_bidsmodel, Wang_asksmodel, Wang_impliedliquiditysurface, sensitivity_Wang_bidsmodel, sensitivity_Wang_asksmodel = bid_ask_cali(real_data, vol_sabr_model)



#BID
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real Mid-prices
    mid_implvola = finsurf_iV[i]

    #Real bids IV 
    y1_points =get_bids(real_data)[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)

    #Wang bid IV
    y5_points = Wang_bidsmodel[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)

    plt.subplots(figsize=(7, 5))
    plt.plot(x_points, mid_implvola, '.', label = "Mid-price IV", color = 'red')
    plt.plot(x_points, y1_implvola, '1', label = "Market bids IV", color = 'orange')
    plt.plot(x_points, y5_implvola, '1', label = "SABR Wang bid IV", color = 'green', alpha = 0.5)
    plt.title('Calibrated bids IV (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('IV')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\sabr_bid_cali_pureIV_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()

    #Create Errror list
    error_bid_IV_wang = np.array(y1_implvola) - np.array(y5_implvola)
    # print('error_bid_IV_minmaxvar')
    # using subplot function and creating
    # plot one
    plt.subplots(figsize=(7, 5))
    plt.plot(x_points, error_bid_IV_wang, label = "SABR Wang IV error", color = 'darkblue')
    plt.title('Bid IV Error (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('Error')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\sabr_bids_cali_pureIV_error_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()


#ASK
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]
    #Real Mid-prices
    mid_implvola = finsurf_iV[i]
    #Real Asks IV 
    y1_points = get_asks(real_data)[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)
    #Wang ASK IV
    y5_points =  Wang_asksmodel[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)

    plt.subplots(figsize=(7, 5))
    plt.plot(x_points, mid_implvola, '.', label = "Mid-price IV", color = 'red')
    plt.plot(x_points, y1_implvola, '1', label = "Market asks IV", color = 'orange')
    plt.plot(x_points, y5_implvola, '1', label = "SABR Wang asks IV", color = 'green', alpha = 0.5)
    plt.title('Calibrated asks IV (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('IV')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\sabr_asks_cali_pureIV_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()

    #Create Errror list
    error_ask_IV_wang = np.array(y1_implvola) - np.array(y5_implvola)
    # print('error_bid_IV_minmaxvar')
    plt.subplots(figsize=(7, 5))
    # using subplot function and creating
    # plot one
    plt.plot(x_points, error_ask_IV_wang, label = "SABR Wang IV error", color = 'darkblue')
    plt.title('Ask IV Error (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('Error')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\sabr_asks_cali_pureIV_error_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()


