from distutils.log import error
from turtle import color
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from py_vollib.black_scholes.implied_volatility import implied_volatility as implVola

from get_data import get_spot
from get_data import get_maturities
from get_data import get_strikes
from get_data import get_spot
from get_data import get_bids
from get_data import get_asks

#Name of Data file
real_data = 'Cali_6mat20k'
#Name of Cali file
calibration_name = 'SP500_calibeta1' 

#Get the current parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))
#File path of cali pickle
pickle_lsv =  parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\finsurf.pkl'
#File path for finsurf instance
finsurf_path = parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\histData.npz'
#File path for finsurf instance
bidask_path = parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\bidaskCaliData.npz'
#READ AND PRINT NPZ FILE
data = np.load(finsurf_path)
data_bidask = np.load(bidask_path)

#DATA FOR GRAPHS BELOW
#PRICES_DATA
prices_data = data['p_data']
#PRICES_DATA
prices_model = data['p_model']
#IV_DATA
IV_data = data['iV_data']
#IV_MODEL
IV_model  = data['iV_model']
#MATURITIES
Mat = get_maturities(real_data)
#STRIKES
K = get_strikes(real_data)
#SPOT
S0 = get_spot(real_data)
#Get Bids Data
Bids = get_bids(real_data)
#Get Asks
Asks  = get_asks(real_data)

#Bids after Cali
minmaxvar_bids = data_bidask['minmaxvar_bidsmodel_MK']
wang_bids = data_bidask['Wang_bidsmodel_MK']

#Asks after Cali
minmaxvar_asks = data_bidask['minmaxvar_asksmodel_MK']
wang_asks = data_bidask['Wang_asksmodel_MK']

#Implied liquidity after cali
IL_minmaxvar = data_bidask['minmaxvar_impliedliquiditysurface_MK']
IL_wang = data_bidask['Wang_impliedliquiditysurface_MK']

#Bids after Cali for new distorion functions 
t_minmaxWang_bidsmodel = data_bidask['t_minmaxWang_bidsmodel_MK']
comp_Wangminmax_bidsmodel = data_bidask['comp_Wangminmax_bidsmodel_MK']
comp_minmaxWang_bidsmodel = data_bidask['comp_minmaxWang_bidsmodel_MK']

#Asks after Cali for new distortion functions
t_minmaxWang_asksmodel = data_bidask['t_minmaxWang_asksmodel_MK']
comp_Wangminmax_asksmodel = data_bidask['comp_Wangminmax_asksmodel_MK']
comp_minmaxWang_asksmodel = data_bidask['comp_minmaxWang_asksmodel_MK']

#t-parameters for new hybrid distortion functions 
t_parameter_minmaxWang  = data_bidask['t_minmaxWang_MK']

#Implied liquidity obtained by new distortion functions
t_minmaxWang_impliedliquiditysurface = data_bidask['t_minmaxWang_impliedliquiditysurface_MK']
comp_Wangminmaxvar_impliedliquiditysurface = data_bidask['comp_Wangminmaxvar_impliedliquiditysurface_MK']
comp_minmaxWang_impliedliquiditysurface = data_bidask['comp_minmaxWang_impliedliquiditysurface_MK']

#Sensitivity parametrs 
#bids
sensitivity_minmaxvar_bidsmodel_MK = data_bidask['sensitivity_minmaxvar_bidsmodel_MK']
sensitivity_Wang_bidsmodel_MK = data_bidask['sensitivity_Wang_bidsmodel_MK']
sensitivity_t_minmaxWang_bidsmodel_MK = data_bidask['sensitivity_t_minmaxWang_bidsmodel_MK']
sensitivity_comp_Wangminmax_bidsmodel_MK = data_bidask['sensitivity_comp_Wangminmax_bidsmodel_MK']
sensitivity_comp_minmaxWang_bidsmodel_MK = data_bidask['sensitivity_comp_minmaxWang_bidsmodel_MK']

#asks
sensitivity_minmaxvar_asksmodel_MK = data_bidask['sensitivity_minmaxvar_asksmodel_MK']
sensitivity_Wang_asksmodel_MK = data_bidask['sensitivity_Wang_asksmodel_MK']
sensitivity_t_minmaxWang_asksmodel_MK = data_bidask['sensitivity_t_minmaxWang_asksmodel_MK']
sensitivity_comp_Wangminmax_asksmodel_MK = data_bidask['sensitivity_comp_Wangminmax_asksmodel_MK']
sensitivity_comp_minmaxWang_asksmodel_MK = data_bidask['sensitivity_comp_minmaxWang_asksmodel_MK']


# np.savez(parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\'+'bidaskCaliData',  mean_payoffs_MK = mean_payoffs_MK, 
#              minmaxvar_impliedliquiditysurface_MK = minmaxvar_impliedliquiditysurface_MK, minmaxvar_bidsmodel_MK = minmaxvar_bidsmodel_MK,
#              sensitivity_minmaxvar_bidsmodel_MK = sensitivity_minmaxvar_bidsmodel_MK, minmaxvar_asksmodel_MK = minmaxvar_asksmodel_MK, sensitivity_minmaxvar_asksmodel_MK = sensitivity_minmaxvar_asksmodel_MK, Wang_impliedliquiditysurface_MK =  Wang_impliedliquiditysurface_MK,
#              Wang_bidsmodel_MK = Wang_bidsmodel_MK, sensitivity_Wang_bidsmodel_MK = sensitivity_Wang_bidsmodel_MK, Wang_asksmodel_MK = Wang_asksmodel_MK, sensitivity_Wang_asksmodel_MK = sensitivity_Wang_asksmodel_MK, t_minmaxWang_impliedliquiditysurface_MK =  t_minmaxWang_impliedliquiditysurface_MK, 
#              t_minmaxWang_bidsmodel_MK = t_minmaxWang_bidsmodel_MK , sensitivity_t_minmaxWang_bidsmodel_MK=sensitivity_t_minmaxWang_bidsmodel_MK, t_minmaxWang_asksmodel_MK = t_minmaxWang_asksmodel_MK, 
#              sensitivity_t_minmaxWang_asksmodel_MK = sensitivity_t_minmaxWang_asksmodel_MK, t_minmaxWang_MK = t_minmaxWang_MK, comp_Wangminmaxvar_impliedliquiditysurface_MK  = comp_Wangminmaxvar_impliedliquiditysurface_MK,
#              comp_Wangminmax_bidsmodel_MK  = comp_Wangminmax_bidsmodel_MK, sensitivity_comp_Wangminmax_bidsmodel_MK = sensitivity_comp_Wangminmax_bidsmodel_MK,
#              comp_Wangminmax_asksmodel_MK=comp_Wangminmax_asksmodel_MK, sensitivity_comp_Wangminmax_asksmodel_MK=sensitivity_comp_Wangminmax_asksmodel_MK, comp_minmaxWang_impliedliquiditysurface_MK =  comp_minmaxWang_impliedliquiditysurface_MK,
#              comp_minmaxWang_bidsmodel_MK = comp_minmaxWang_bidsmodel_MK, sensitivity_comp_minmaxWang_bidsmodel_MK=sensitivity_comp_minmaxWang_bidsmodel_MK,
#              comp_minmaxWang_asksmodel_MK = comp_minmaxWang_asksmodel_MK, sensitivity_comp_minmaxWang_asksmodel_MK =  sensitivity_comp_minmaxWang_asksmodel_MK)





#Plot the implied volatility 
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
# defining surface and axes
y = np.array(get_maturities(real_data))
x = moneyness[0]
x, y = np.meshgrid(x, y)
z1 = data['iV_data']
fig = plt.figure(figsize=(7, 5))
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
# syntax for plotting
ax.plot_surface(x, y, z1, cmap ='viridis', edgecolor ='green')
ax.set_title('Implied volatility surface')
ax.set_ylabel('Time to maturity (years)')
ax.set_xlabel('Moneyness')
# plt.show()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_volatility_surface.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.close()


#Plot the implied volatility squared error after Calibration
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
# defining surface and axes
y = np.array(get_maturities(real_data))
x = moneyness[0]
x, y = np.meshgrid(x, y)
z1 = data['iV_data']
z2 = data['iV_model']
z = (z1 - z2)**2
fig = plt.figure(figsize=(7, 5))
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
# syntax for plotting
ax.plot_surface(x, y, z, cmap ='viridis', edgecolor ='yellow')
ax.set_title('Implied volatility squared error')
ax.set_ylabel('Time to maturity (years)')
ax.set_xlabel('Moneyness')
name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_volatility_squared_error_surface.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
# plt.show()
plt.close()

#Plot implied volatility model vs data
for i in range(len(Mat)):
    x_points = moneyness[i]
    y1_points = IV_data[i]
    y2_points = IV_model[i]
    error_iv =  y1_points - y2_points
    

    # print('The error average absolute error for maturity {}'.format(Mat[i]), np.mean(np.abs(error_iv)))
    # print('MAX ERROR', np.max(np.abs(error_iv)))
    # print('INDEX MAX ERROR', list(np.abs(error_iv)).index(np.max(np.abs(error_iv))))

    #Figure of Smiles
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, y1_points, label = 'Data implied vol')
    plt.plot(x_points, y2_points, label = 'LSV-SABR fit implied vol')
    plt.ylabel('Implied Volatility')
    plt.xlabel('Moneyness')
    plt.title('Volatility smile (T = {})'.format(Mat[i]))
    plt.legend()
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\LSV_SABR_implied_vol_smile_mat{}.eps"
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

    #Fig Error 
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, error_iv)
    plt.ylabel('Error')
    plt.xlabel('Moneyness')
    plt.title('Implied volatility error (T = {})'.format(Mat[i]))
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_volatility_error_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

#Plot Bid Prices for ATM options
#MINMAXVAR
#X axis is the same
x_points = Mat
y1_points = [i[12] for i in prices_data]
y2_points = [i[12] for i in Bids]
y3_points = [i[12] for i in minmaxvar_bids]
plt.figure(figsize=(7, 5))
plt.plot(x_points, y1_points, '.', label = "Mid-price", color = 'red')
plt.plot(x_points, y2_points, '1', label = "Market bids", color = 'orange')
plt.plot(x_points, y3_points, '1', label = "Calibrated bids", color = 'blue', alpha = 0.5)
plt.title('Minmax distortion bid prices (ATM)')
plt.xlabel('Maturity')
plt.ylabel('Price')
plt.legend()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_atm_cali_prices_minmax.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.close()

#WANG
#X axis is the same
x_points = Mat
y1_points = [i[12] for i in prices_data]
y2_points = [i[12] for i in Bids]
y3_points = [i[12] for i in wang_bids]
plt.figure(figsize=(7, 5))
plt.plot(x_points, y1_points, '.', label = "Mid-price", color = 'red')
plt.plot(x_points, y2_points, '1', label = "Market bids", color = 'orange')
plt.plot(x_points, y3_points, '1', label = "Calibrated bids", color = 'blue', alpha = 0.5)
plt.title('Wang distortion bid prices (ATM)')
plt.xlabel('Maturity')
plt.ylabel('Price')
plt.legend()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_atm_cali_prices_wang.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.close()

#Plot Ask Prices for ATM options
#MINMXVAR
#X axis is the same
x_points = Mat
y1_points = [i[12] for i in prices_data]
y2_points = [i[12] for i in Asks]
y3_points = [i[12] for i in minmaxvar_asks]
plt.figure(figsize=(7, 5))
plt.plot(x_points, y1_points, '.', label = "Mid-price", color = 'red')
plt.plot(x_points, y2_points, '2', label = "Market asks", color = 'orange')
plt.plot(x_points, y3_points, '2', label = "Calibrated asks", color = 'blue', alpha = 0.5)
plt.title('Minmax distortion ask prices (ATM)')
plt.xlabel('Maturity')
plt.ylabel('Price')
plt.legend()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_atm_cali_prices_minmax.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
plt.close()

#WANG
#X axis is the same
x_points = Mat
y1_points = [i[12] for i in prices_data]
y2_points = [i[12] for i in Asks]
y3_points = [i[12] for i in wang_asks]
plt.figure(figsize=(7, 5))
plt.plot(x_points, y1_points, '.', label = "Mid-price", color = 'red')
plt.plot(x_points, y2_points, '2', label = "Market asks", color = 'orange')
plt.plot(x_points, y3_points, '2', label = "Calibrated asks", color = 'blue', alpha = 0.5)
plt.title('Wang distortion ask prices (ATM)')
plt.xlabel('Maturity')
plt.ylabel('Price')
plt.legend()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_atm_cali_prices_wang.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')
# plt.show()
plt.close()

#Plot Bid and Ask IV
#BID
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real Mid-prices
    mid_prices = prices_data
    mid_implvola = IV_data[i]

    #Real bids IV 
    y1_points = Bids[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)

    #Minmax bid IV 
    y3_points = minmaxvar_bids[i]
    y3_implvola = []
    for x in range(len(y3_points)):
        y = implVola(y3_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y3_implvola.append(y)

    #Wang bid IV
    y5_points = wang_bids[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)

    plt.subplots(figsize=(7, 5))
    plt.plot(x_points, mid_implvola, '.', label = "Mid-price IV", color = 'red')
    plt.plot(x_points, y1_implvola, '1', label = "Market bids IV", color = 'orange')
    plt.plot(x_points, y3_implvola, '1', label = "Minmax bid IV", color = 'blue', alpha = 0.5)
    plt.plot(x_points, y5_implvola, '1', label = "Wang bid IV", color = 'green', alpha = 0.5)
    plt.title('Calibrated bids IV (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('IV')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_cali_pureIV_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()


#ASK
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real Mid-prices
    mid_prices = prices_data
    mid_implvola = IV_data[i]

    #Real bids IV 
    y1_points = Asks[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)

    #Minmax bid IV 
    y3_points = minmaxvar_asks[i]
    y3_implvola = []
    for x in range(len(y3_points)):
        y = implVola(y3_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y3_implvola.append(y)

    #Wang bid IV
    y5_points = wang_asks[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)

    plt.subplots(figsize=(7, 5))
    plt.plot(x_points, mid_implvola, '.', label = "Mid-price IV", color = 'red')
    plt.plot(x_points, y1_implvola, '1', label = "Market asks IV", color = 'orange')
    plt.plot(x_points, y3_implvola, '1', label = "Minmax asks IV", color = 'blue', alpha = 0.5)
    plt.plot(x_points, y5_implvola, '1', label = "Wang asks IV", color = 'green', alpha = 0.5)
    plt.title('Calibrated asks IV (T = {})'.format(Mat[i]), fontsize = 12)
    plt.xlabel('Moneyness')
    plt.ylabel('IV')
    plt.legend()
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\asks_cali_pureIV_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.close()


sum_minmax_bid_error = []
sum_wang_bid_error = []
#Plot implied vola error Bids
for i in range(0,len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real bids IV 
    y1_points = Bids[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)

    #Minmax bid IV 
    y3_points = minmaxvar_bids[i]
    y3_implvola = []
    for x in range(len(y3_points)):
        y = implVola(y3_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y3_implvola.append(y)

    #Wang bid
    y5_points = wang_bids[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)


    #Create Errror list
    error_bid_IV_minmaxvar = np.array(y1_implvola) - np.array(y3_implvola)
    error_bid_IV_wang = np.array(y1_implvola) - np.array(y5_implvola)
    # print('error_bid_IV_minmaxvar')
    ##ASK ERROR
    sum_minmax_bid_error.append(np.sum(np.abs(error_bid_IV_minmaxvar)))
    sum_wang_bid_error.append(np.sum(np.abs(error_bid_IV_wang )))


    plt.subplots(figsize=(14, 5))
    # using subplot function and creating
    # plot one
    plt.subplot(1, 2, 1)
    plt.plot(x_points, error_bid_IV_minmaxvar, label = "IV error", color = 'darkblue')
    plt.title('Minmax distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('Error')
    plt.legend()

    # using subplot function and creating plot two
    plt.subplot(1, 2, 2)
    plt.plot(x_points, error_bid_IV_wang, label = "IV error", color = 'darkblue')
    plt.title('Wang distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('Error')
    plt.legend()
    # space between the plots
    # plt.tight_layout()
    # show plot
    plt.suptitle('Bid calibration IV error(T = {})'.format(Mat[i]), fontsize = 12)
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_cali_IVerror_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

print("The Average bid error of minmaxvar function ", np.sum(sum_minmax_bid_error)/120)
print("The Average bid error of wang function", np.sum(sum_wang_bid_error)/120)



sum_minmax_ask_error = []
sum_wang_ask_error = []
#Plot implied vola error ask
for i in range(0,len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real bids IV 
    y2_points = Asks[i]
    y2_implvola = []
    for x in range(len(y2_points)):
        y = implVola(y2_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y2_implvola.append(y)

    #Minmax bid IV 
    y4_points = minmaxvar_asks[i]
    y4_implvola = []
    for x in range(len(y4_points)):
        y = implVola(y4_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y4_implvola.append(y)

    #Wang bid-ask
    y6_points = wang_asks[i]
    y6_implvola = []
    for x in range(len(y6_points)):
        y = implVola(y6_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y6_implvola.append(y)


    #Create Errror list
    error_ask_IV_minmaxvar = np.array(y2_implvola) - np.array(y4_implvola)
    error_ask_IV_wang = np.array(y2_implvola) - np.array(y6_implvola)

    ##ASK ERROR
    sum_minmax_ask_error.append(np.sum(np.abs(error_ask_IV_minmaxvar)))
    sum_wang_ask_error.append(np.sum(np.abs(error_ask_IV_wang )))

    #
    plt.subplots(figsize=(14, 5))
    # using subplot function and creating
    # plot one
    plt.subplot(1, 2, 1)
    plt.plot(x_points, error_ask_IV_minmaxvar, label = "IV error", color = 'darkblue')
    plt.title('Minmax distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()

    # using subplot function and creating plot two
    plt.subplot(1, 2, 2)
    plt.plot(x_points, error_ask_IV_wang, label = "IV error", color = 'darkblue')
    plt.title('Wang distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()
    # space between the plots
    # plt.tight_layout()
    # show plot
    plt.suptitle('Ask calibration IV error(T = {})'.format(Mat[i]), fontsize = 12)
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_cali_IVerror_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()




print("The Average ask error of minmaxvar function ", np.sum(sum_minmax_ask_error)/120)
print("The Average ask error of wang function", np.sum(sum_wang_ask_error)/120)


##############  T-Minmax-Wang ##################
sum_t_bid_error = []
sum_t_ask_error = []

for i in range(len(Mat)):
    x_points = moneyness[i]
    ###Load Real Bids and Asks (Caculate their respective implied volatilies)
    #Real bids IV 
    y1_points = Bids[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)
    #Real asks IV 
    y2_points = Asks[i]
    y2_implvola = []   
    for x in range(len(y2_points)):
        y = implVola(y2_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y2_implvola.append(y)
    
    #Load Model Bid and Asks (t) 
    #Bid 
    y3_points = t_minmaxWang_bidsmodel[i]
    y3_implvola = []
    for x in range(len(y3_points)):
        y = implVola(y3_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y3_implvola.append(y)
    #Ask
    y4_points = t_minmaxWang_asksmodel[i]
    y4_implvola = []
    for x in range(len(y4_points)):
        y = implVola(y4_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y4_implvola.append(y)

    #Calculate Errors
    bid_error_iv_t_minmaxwang =   np.array(y1_implvola) -  np.array(y3_implvola)
    ask_error_iv_t_minmaxwang =  np.array(y2_implvola) -  np.array(y4_implvola)

    sum_t_bid_error.append(np.sum(np.abs(bid_error_iv_t_minmaxwang)))
    sum_t_ask_error.append(np.sum(np.abs(ask_error_iv_t_minmaxwang)))

    #Load t-parameter
    t_parameter_minmaxWang_mat = t_parameter_minmaxWang[i]

    #Fig Error Bid
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, bid_error_iv_t_minmaxwang, label = "IV error", color = 'darkblue')
    plt.ylabel('Error')
    plt.xlabel('Moneyness')
    plt.title('Bid Implied volatility error (T = {})'.format(Mat[i]))
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_error_iv_t_minmaxwang_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

    #Fig Error Ask 
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, ask_error_iv_t_minmaxwang, label = "IV error", color = 'darkblue')
    plt.ylabel('Error')
    plt.xlabel('Moneyness')
    plt.title('Ask Implied volatility error (T = {})'.format(Mat[i]))
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_error_iv_t_minmaxwang_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

    #Fig T-parameter
    plt.figure(figsize=(7, 5))
    plt.ylim(-0.05, 1.05)
    plt.scatter(x_points, t_parameter_minmaxWang_mat)
    plt.ylabel('Error')
    plt.xlabel('Moneyness')
    plt.title('t-parameter for maturity (T = {})'.format(Mat[i]))
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\t_parameter_minmaxWang_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()


# #Asks after Cali for new distortion functions
# t_minmaxWang_asksmodel = data_bidask['t_minmaxWang_asksmodel_MK']
# comp_Wangminmax_asksmodel = data_bidask['comp_Wangminmax_asksmodel_MK']
# comp_minmaxWang_asksmodel = data_bidask['comp_minmaxWang_asksmodel_MK']

# #t-parameters for new hybrid distortion functions 
# t_parameter_minmaxWang  = data_bidask['t_minmaxWang_MK']
print("The Average bid error of new T-minmaxvar function", np.sum(sum_t_bid_error)/120)
print("The Average ask error of new T-minmaxvar function", np.sum(sum_t_ask_error)/120)




sum_t_bid_error = []
sum_t_ask_error = []

####################Composition of two function########### 

sum_bid_error_minmaxwang = []
sum_bid_error_wangminmax = []
#Plot implied vola error bid
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real bids IV 
    y1_points = Bids[i]
    y1_implvola = []
    for x in range(len(y1_points)):
        y = implVola(y1_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y1_implvola.append(y)

    #Minmax bid IV 
    y3_points = comp_Wangminmax_bidsmodel[i]
    y3_implvola = []
    for x in range(len(y3_points)):
        y = implVola(y3_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y3_implvola.append(y)

    #Wang bid-ask
    y5_points = comp_minmaxWang_bidsmodel[i]
    y5_implvola = []
    for x in range(len(y5_points)):
        y = implVola(y5_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y5_implvola.append(y)


    #Create Errror list
    # print("Minmax-wang IV", y3_implvola)
    # print("Wang-Minmax IV", y5_implvola)
    #Error Minmax-Wang
    error_ask_IV_Minmaxvar_Wang = np.array(y1_implvola) - np.array(y3_implvola)
    
    #Error Wang-Minmaxvar
    error_ask_IV_Wang_Minmaxvar = np.array(y1_implvola) - np.array(y5_implvola)

    #
    sum_bid_error_minmaxwang.append(np.sum(np.abs(error_ask_IV_Minmaxvar_Wang )))
    sum_bid_error_wangminmax.append(np.sum(np.abs(error_ask_IV_Wang_Minmaxvar)))

    plt.subplots(figsize=(14, 5))
    # using subplot function and creating
    # plot one
    plt.subplot(1, 2, 1)
    plt.plot(x_points, error_ask_IV_Minmaxvar_Wang, label = "IV error", color = 'darkblue')
    plt.title('The Minmaxvar-Wang distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()

    # using subplot function and creating plot two
    plt.subplot(1, 2, 2)
    plt.plot(x_points, error_ask_IV_Wang_Minmaxvar, label = "IV error", color = 'darkblue')
    plt.title('The Wang-Minmaxvar distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()
    # space between the plots
    # plt.tight_layout()
    # show plot
    plt.suptitle('Bid calibration IV error(T = {})'.format(Mat[i]), fontsize = 12)
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_cali_comp_IVerror_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()
    plt.close()

print("The Average bid error of new Minmaxvar-Wang function", np.sum(sum_bid_error_minmaxwang)/120)
print("The Average bid error of new Wang-Minmaxvar function", np.sum(sum_bid_error_wangminmax)/120)

sum_ask_error_minmaxwang = []
sum_ask_error_wangminmax = []
#Plot implied vola error ask for composition of functions
for i in range(len(Mat)):
    #X axis is the same
    x_points = moneyness[i]

    #Real asks IV 
    y2_points = Asks[i]
    y2_implvola = []
    
    for x in range(len(y2_points)):
        y = implVola(y2_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y2_implvola.append(y)

    #Minmax-Wang ask IV 
    y4_points = comp_Wangminmax_asksmodel[i]
    y4_implvola = []
    for x in range(len(y4_points)):
        y = implVola(y4_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y4_implvola.append(y)

    #Wang-Minmaxvar ask IV
    y6_points = comp_minmaxWang_asksmodel[i]
    y6_implvola = []
    for x in range(len(y6_points)):
        y = implVola(y6_points[x], S0, K[i][x], Mat[i], 0.0, 'c')
        y6_implvola.append(y)


    #Create Error list
    #Error Minmax-Wang
    error_ask_IV_Minmaxvar_Wang = np.array(y2_implvola) - np.array(y4_implvola)
    
    #Error Wang-Minmaxvar
    error_ask_IV_Wang_Minmaxvar = np.array(y2_implvola) - np.array(y6_implvola)

    ###
    sum_ask_error_minmaxwang.append(np.sum(np.abs(error_ask_IV_Minmaxvar_Wang )))
    sum_ask_error_wangminmax.append(np.sum(np.abs(error_ask_IV_Wang_Minmaxvar)))
    #
    plt.subplots(figsize=(14, 5))
    # using subplot function and creating
    # plot one
    plt.subplot(1, 2, 1)
    plt.plot(x_points, error_ask_IV_Minmaxvar_Wang, label = "IV error", color = 'darkblue')
    plt.title('The Minmaxvar-Wang distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()

    # using subplot function and creating plot two
    plt.subplot(1, 2, 2)
    plt.plot(x_points, error_ask_IV_Wang_Minmaxvar, label = "IV error", color = 'darkblue')
    plt.title('The Wang-Minmaxvar distortion')
    plt.xlabel('Moneyness')
    plt.ylabel('IV error')
    plt.legend()
    # space between the plots
    # plt.tight_layout()
    # show plot
    plt.suptitle('Ask calibration IV error(T = {})'.format(Mat[i]), fontsize = 12)
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_cali_comp_IVerror_mat{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    plt.show()
    # plt.close()

print("The Average ask error of new Minmaxvar-Wang function", np.sum(sum_ask_error_minmaxwang)/120)
print("The Average ask error of new Wang-Minmaxvar function", np.sum(sum_ask_error_wangminmax)/120)

#####PLOT THE IMPLIED LIQUIDITY

#Plot Implied liquidity Minmax
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
# defining surface and axes
y = np.array(get_maturities(real_data))
x = moneyness[0]
x, y = np.meshgrid(x, y)
z1 = IL_minmaxvar
fig = plt.figure(figsize=(7, 5))
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
# syntax for plotting
ax.plot_surface(x, y, z1, cmap ='viridis', edgecolor ='blue')
ax.set_title('Implied liquidity surface (minmaxvar distortion)')
ax.set_ylabel('Time to maturity (years)')
ax.set_xlabel('Moneyness')
# plt.show()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_liquidity_surface_minmaxvar.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')


#Plot Implied liquidity Wang transform
moneyness = np.array(get_strikes(real_data))/get_spot(real_data)
# defining surface and axes
y = np.array(get_maturities(real_data))
x = moneyness[0]
x, y = np.meshgrid(x, y)
z1 = IL_wang
fig = plt.figure(figsize=(7, 5))
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
# syntax for plotting
ax.plot_surface(x, y, z1, cmap ='viridis', edgecolor ='blue')
ax.set_title('Implied liquidity surface (wang distortion)')
ax.set_ylabel('Time to maturity (years)')
ax.set_xlabel('Moneyness')
# plt.show()
name_fig = 'PastCalibrations\\'  + calibration_name + "\\implied_liquidity_surface_wang.eps"
plt.savefig(name_fig, dpi=1200, format = 'eps')




#Plot the sensitivity parameters
##Get implied Volatility
            # implVola(model_price[iter], self.spot, K, mat, 0.0, 'c')


#Sensitivity parametrs 

#bids
sensitivity_minmaxvar_bidsmodel_MK = data_bidask['sensitivity_minmaxvar_bidsmodel_MK']
sensitivity_Wang_bidsmodel_MK = data_bidask['sensitivity_Wang_bidsmodel_MK']
sensitivity_t_minmaxWang_bidsmodel_MK = data_bidask['sensitivity_t_minmaxWang_bidsmodel_MK']

#asks
sensitivity_minmaxvar_asksmodel_MK = data_bidask['sensitivity_minmaxvar_asksmodel_MK']
sensitivity_Wang_asksmodel_MK = data_bidask['sensitivity_Wang_asksmodel_MK']
sensitivity_t_minmaxWang_asksmodel_MK = data_bidask['sensitivity_t_minmaxWang_asksmodel_MK']



############## Sensitivity Parameters ##################
for i in range(len(Mat)):
    #X points
    x_points = moneyness[i]
    #Bid Sensitivities
    s_bid_minmax = sensitivity_minmaxvar_bidsmodel_MK[i]
    s_bid_wang = sensitivity_Wang_bidsmodel_MK[i]
    s_bid_tminmaxwang = sensitivity_t_minmaxWang_bidsmodel_MK[i]
    #Ask Sensitivites
    s_ask_minmax = sensitivity_minmaxvar_asksmodel_MK[i]
    s_ask_wang =sensitivity_Wang_asksmodel_MK[i]
    s_ask_tminmaxwang = sensitivity_t_minmaxWang_asksmodel_MK[i]

    #Fig Error Bid
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, s_bid_minmax, label = "Minmaxvar", color = 'blue')
    plt.plot(x_points, s_bid_wang, label = "Wang", color = 'orange')
    plt.plot(x_points, s_bid_tminmaxwang, label = "t-Minmax-Wang", color = 'green')
    plt.ylabel('Sensitivity')
    plt.xlabel('Moneyness')
    plt.title('Sensitivity Parameters bid (T = {})'.format(Mat[i]))
    plt.legend()
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\bid_error_sensitivities{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()

    #Fig Error Ask 
    plt.figure(figsize=(7, 5))
    plt.plot(x_points, s_ask_minmax, label = "Minmaxvar", color = 'blue')
    plt.plot(x_points, s_ask_wang, label = "Wang", color = 'orange')
    plt.plot(x_points, s_ask_tminmaxwang, label = "t-Minmax-Wang", color = 'green')
    plt.ylabel('Sensitivity')
    plt.xlabel('Moneyness')
    plt.title('Sensitivity Parameters ask (T = {})'.format(Mat[i]))
    plt.legend()
    #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    name_fig = 'PastCalibrations\\'  + calibration_name + "\\ask_error_sensitivities{}.eps".format(Mat[i])
    plt.savefig(name_fig, dpi=1200, format = 'eps')
    # plt.show()


    #Fig T-parameter
    # plt.figure(figsize=(7, 5))
    # plt.ylim(-0.05, 1.05)
    # plt.scatter(x_points, t_parameter_minmaxWang_mat)
    # plt.ylabel('Error')
    # plt.xlabel('Moneyness')
    # plt.title('t-parameter for maturity (T = {})'.format(Mat[i]))
    # #print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
    # name_fig = 'PastCalibrations\\'  + calibration_name + "\\t_parameter_minmaxWang_mat{}.eps".format(Mat[i])
    # plt.savefig(name_fig, dpi=1200, format = 'eps')
    # # plt.show()
    # plt.close()