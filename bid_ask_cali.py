from statistics import mean
from unittest import result
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import pickle
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import norm

from get_data import get_maturities
from get_data import get_strikes
from get_data import get_spot
from get_data import get_bids
from get_data import get_asks

#Name of Data file
real_data = "Cali_6mat20k"
#Name of Cali file
calibration_name = 'SP500_Calibeta1'
# get the parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))
#File path for finsurf instance
finsurf_path = parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\histData.npz'
#READ NPZ FILE
data = np.load(finsurf_path)


#Function to get MC Payoffs in array of  size (M, K), where M is maturities and K the strikes
def get_mc_payoffs(real_data, data):
    K = get_strikes(real_data)
    N = len(data['mc_prices_model'][0])
    payoffs_MK = []
    mean_payoffs_MK = []
    for i in range(len(data['mc_prices_model'])):
        payoffs_K = []
        mean_payoffs_K = []
        for k in range(len(K[0])):
            listk = np.array([K[i][k]]*N)
            Sminusk = np.subtract(data['mc_prices_model'][i], listk)
            list0 = np.array([0]*N)
            payoff = np.maximum(Sminusk, list0)
            mean_payoff = np.mean(payoff)
            #Append payoff for strike K_i 
            payoffs_K.append(payoff)
            mean_payoffs_K.append(mean_payoff)
        #Append all payoffs for maturity M_i  
        payoffs_MK.append(payoffs_K)
        mean_payoffs_MK.append(mean_payoffs_K)
    return payoffs_MK, mean_payoffs_MK

#Defining Important function later used to generate bids and asks using minmaxvar distorsion function
#import numexpr as ne
# https://github.com/khrapovs/fangoosterlee/blob/master/fangoosterlee/cosmethod.py



#Define min max function and how to derive the bids from such function
def minmaxvar(l, u):
    result = 1 - (1 - u**(1/(l+1)))**(1+l)
    return result
#give bid and ask prices for the model under this distrortion function
def minmaxvar_bid_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v2 = np.arange(0, N) / N
    psi1 = minmaxvar(l, v1)
    psi2 = minmaxvar(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def minmaxvar_ask_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v1 = v1[::-1]
    v2 = np.arange(0, N) / N
    v2 = v2[::-1] 
    psi1 = minmaxvar(l, v1)
    psi2 = minmaxvar(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result
#Error function for the calibration functional
def minmaxvar_sumofsquareserror(l, payoff, real_bid, real_ask):
    askmodel  = minmaxvar_ask_model(payoff, l) 
    bidmodel = minmaxvar_bid_model(payoff, l)
    return (askmodel - real_ask)**2  + (bidmodel - real_bid)**2

#Create necessary functions for Wang distorsion function
def Wang(l, u):
    result = norm.cdf(ndtri(u)+l)
    return result
#Get bid ask prices for this distortion
def Wang_bid_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v2 = np.arange(0, N) / N
    psi1 = Wang(l, v1)
    psi2 = Wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def Wang_ask_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v1 = v1[::-1]
    v2 = np.arange(0, N) / N
    v2 = v2[::-1] 
    psi1 = Wang(l, v1)
    psi2 = Wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def Wang_sumofsquareserror(l, payoff, real_bid, real_ask):
    askmodel  = Wang_ask_model(payoff, l) 
    bidmodel = Wang_bid_model(payoff, l)
    return (askmodel - real_ask)**2  + (bidmodel - real_bid)**2

#Get bid and ask prices for the combination of two distortion functions
def t_minmax_wang_bid_model(mc_payoffs, l, t):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v2 = np.arange(0, N) / N
    psi1 = t*(minmaxvar(l, v1))+(1-t)*Wang(l, v1)
    psi2 = t*(minmaxvar(l, v2))+(1-t)*Wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def t_minmax_wang_ask_model(mc_payoffs, l, t):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v1 = v1[::-1]
    v2 = np.arange(0, N) / N
    v2 = v2[::-1] 
    psi1 = t*(minmaxvar(l, v1))+(1-t)*Wang(l, v1)
    psi2 = t*(minmaxvar(l, v2))+(1-t)*Wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

#Error for function of linear combination of two distortion functions
def minmax_wang_sumofsqurederror(t_lambda, payoff, real_bid, real_ask):
    t, l = t_lambda 
    askmodel  =  t_minmax_wang_ask_model(payoff, l, t)
    bidmodel =  t_minmax_wang_bid_model(payoff, l, t)
    return (askmodel - real_ask)**2  + (bidmodel - real_bid)**2

#Function Defined as the composition of wang and minmavar distortion functions
def comp_wang_minmaxvar(l, u):
    return Wang(l, minmaxvar(l, u))

#Generate new bid and ask for the model under this new distortion function
def comp_wang_minmaxvar_bid_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v2 = np.arange(0, N) / N
    psi1 = comp_wang_minmaxvar(l, v1)
    psi2 = comp_wang_minmaxvar(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def comp_wang_minmaxvar_ask_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v1 = v1[::-1]
    v2 = np.arange(0, N) / N
    v2 = v2[::-1] 
    psi1 = comp_wang_minmaxvar(l, v1)
    psi2 = comp_wang_minmaxvar(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result
#Define the Error for the calibration function
def Comp_Wang_Minmax_sumofsquareserror(l, payoff, real_bid, real_ask):
    askmodel  = comp_wang_minmaxvar_ask_model(payoff, l)

    bidmodel =comp_wang_minmaxvar_bid_model(payoff, l)
    return (askmodel - real_ask)**2  + (bidmodel - real_bid)**2

#Function Defined as the composition of minmavar and minmaxvar distortion functions
def comp_minmaxvar_wang(l, u):
    return minmaxvar(l, Wang(l, u))
    
#Generate bid and ask for the model under this new distortion function
def comp_minmaxvar_wang_bid_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v2 = np.arange(0, N) / N
    psi1 = comp_minmaxvar_wang(l, v1)
    psi2 = comp_minmaxvar_wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

def comp_minmaxvar_wang_ask_model(mc_payoffs, l):
    N = len(mc_payoffs) 
    v1 = np.arange(1, N+1) / N
    v1 = v1[::-1]
    v2 = np.arange(0, N) / N
    v2 = v2[::-1] 
    psi1 = comp_minmaxvar_wang(l, v1)
    psi2 = comp_minmaxvar_wang(l, v2)
    result = np.dot(psi1 - psi2, mc_payoffs)
    return result

#Define error function for the calibration 
def Comp_Minmax_Wang_sumofsquareserror(l, payoff, real_bid, real_ask):
    askmodel  = comp_minmaxvar_wang_ask_model(payoff, l) 
    bidmodel = comp_minmaxvar_wang_bid_model(payoff, l)
    return (askmodel - real_ask)**2  + (bidmodel - real_bid)**2


#Create a function to generate bids and asks from the model
def bid_ask_cali(real_data, payoffs_MK):
    K = get_strikes(real_data)
    #Create object to store minmaxvar results
    minmaxvar_impliedliquiditysurface_MK = []
    minmaxvar_bidsmodel_MK = []
    sensitivity_minmaxvar_bidsmodel_MK = []
    minmaxvar_asksmodel_MK = []
    sensitivity_minmaxvar_asksmodel_MK = []
    #Create object to store Wang results
    Wang_impliedliquiditysurface_MK = []
    Wang_bidsmodel_MK = []
    sensitivity_Wang_bidsmodel_MK = []
    Wang_asksmodel_MK = []
    sensitivity_Wang_asksmodel_MK = []
    #Create Object to store linear combinaion of both method
    t_minmaxWang_impliedliquiditysurface_MK = []
    t_minmaxWang_bidsmodel_MK = []
    sensitivity_t_minmaxWang_bidsmodel_MK = []
    t_minmaxWang_asksmodel_MK = []
    sensitivity_t_minmaxWang_asksmodel_MK = []
    t_minmaxWang_MK =[]
    #Create Object to store composition wang minmaxvar
    comp_Wangminmaxvar_impliedliquiditysurface_MK = []
    comp_Wangminmax_bidsmodel_MK = []
    sensitivity_comp_Wangminmax_bidsmodel_MK = []
    comp_Wangminmax_asksmodel_MK = []
    sensitivity_comp_Wangminmax_asksmodel_MK = []
    #Create Object to store composition minmaxvar wang
    comp_minmaxWang_impliedliquiditysurface_MK = []
    comp_minmaxWang_bidsmodel_MK = []
    sensitivity_comp_minmaxWang_bidsmodel_MK = []
    comp_minmaxWang_asksmodel_MK = []
    sensitivity_comp_minmaxWang_asksmodel_MK = []
    #Create Object to store composition minmaxvar wang
    for i in range(len(data['mc_prices_model'])):
        #Create object to store minmaxvar results
        minmaxvar_impliedliquiditysurface_K = []
        minmaxvar_bidsmodel_K = []
        sensitivity_minmaxvar_bidsmodel_K = []
        minmaxvar_asksmodel_K = []
        sensitivity_minmaxvar_asksmodel_K = []
        #Create object to store Wang results
        Wang_impliedliquiditysurface_K = []
        Wang_bidsmodel_K = []
        sensitivity_Wang_bidsmodel_K = []
        Wang_asksmodel_K = []
        sensitivity_Wang_asksmodel_K = []
        #Create Object to store linear combinaion of both method t0minmacwang
        t_minmaxWang_impliedliquiditysurface_K = []
        t_minmaxWang_bidsmodel_K = []
        sensitivity_t_minmaxWang_bidsmodel_K = []
        t_minmaxWang_asksmodel_K = []
        sensitivity_t_minmaxWang_asksmodel_K = []
        t_minmaxWang_K = []
        #Create Object to store composition wang minmaxvar
        comp_Wangminmaxvar_impliedliquiditysurface_K = []
        comp_Wangminmax_bidsmodel_K = []
        sensitivity_comp_Wangminmax_bidsmodel_K = []
        comp_Wangminmax_asksmodel_K = []
        sensitivity_comp_Wangminmax_asksmodel_K = []
        #Create Object to store composition minmaxvar wang
        comp_minmaxWang_impliedliquiditysurface_K = []
        comp_minmaxWang_bidsmodel_K = []
        sensitivity_comp_minmaxWang_bidsmodel_K = []
        comp_minmaxWang_asksmodel_K = []
        sensitivity_comp_minmaxWang_asksmodel_K = []
        for k in range(len(K[0])):
            print('\n\n Getting bids and ask for maturity {} and strike {}\n\n'.format(i,k))
            #Get the MC payoff and the real bid and ask
            payoff = np.sort(payoffs_MK[i][k])
            real_bid = get_bids(real_data)[i][k]
            real_ask = get_asks(real_data)[i][k]
            #Find the lambdas that minimizes the sum of squared error for each distorsion function
            result_minmaxvar = minimize_scalar(lambda l: minmaxvar_sumofsquareserror(l, payoff = payoff, real_bid = real_bid, real_ask = real_ask), bounds=(0,1), method = 'bounded')
            result_wang =  minimize_scalar(lambda l: Wang_sumofsquareserror(l, payoff = payoff, real_bid = real_bid, real_ask = real_ask), bounds=(0,1), method = 'bounded')
            #Get Lambda and t that minimize the combination of 
            bounds = [(0, 1), (0, 1)]
            initial_guess = [0.5, 0.5]
            x0 = np.array(initial_guess)
            res = minimize(minmax_wang_sumofsqurederror, x0, args = (payoff, real_bid, real_ask) , method='L-BFGS-B', bounds=bounds)
            #t and lammbda
            t, mnv_wang_minlambda = res.x
            #Find the lambdas that minimizes the sum of squared error for each composition of distorsion functions
            result_comp_wang_minmaxvar = minimize_scalar(lambda l: Comp_Wang_Minmax_sumofsquareserror(l, payoff = payoff, real_bid = real_bid, real_ask = real_ask), bounds=(0,1), method = 'bounded')
            result_comp_minmaxvar_wang =  minimize_scalar(lambda l: Comp_Minmax_Wang_sumofsquareserror(l, payoff = payoff, real_bid = real_bid, real_ask = real_ask), bounds=(0,1), method = 'bounded')            
            #Get the lambda Value and the corresponded bid and ask for that lambda value for minmaxvar
            mmv_minlambda = result_minmaxvar.x 
            mmv_bidmodel = minmaxvar_bid_model(payoff, mmv_minlambda)
            mmv_askmodel = minmaxvar_ask_model(payoff, mmv_minlambda)
            #Same for Wang
            wang_minlambda = result_wang.x
            wang_bidmodel = Wang_bid_model(payoff, wang_minlambda)
            wang_askmodel = Wang_ask_model(payoff, wang_minlambda)
            #Get the Corresponding bid and ask for the linear combination of distortion function    
            mnv_wang_bidmodel = t_minmax_wang_bid_model(payoff, mnv_wang_minlambda, t)
            mnv_wang_askmodel = t_minmax_wang_ask_model(payoff, mnv_wang_minlambda , t)
            #Get the lambda Value and the correspondig bid and ask for that lambda value for the composition wang minmaxvar
            comp_wang_minmaxvar_minlambda = result_comp_wang_minmaxvar.x
            comp_wang_minmaxvar_bidmodel = comp_wang_minmaxvar_bid_model(payoff, comp_wang_minmaxvar_minlambda)
            comp_wang_minmaxvar_askmodel = comp_wang_minmaxvar_ask_model(payoff, comp_wang_minmaxvar_minlambda)
            #Same for minmaxvar_wang
            comp_minmaxvar_wang_minlambda = result_comp_minmaxvar_wang.x
            comp_minmaxvar_wang_bidmodel = comp_minmaxvar_wang_bid_model(payoff, comp_minmaxvar_wang_minlambda )
            comp_minmaxvar_wang_askmodel = comp_minmaxvar_wang_ask_model(payoff, comp_minmaxvar_wang_minlambda )
            #Calculate sensitivities of the lambda parameter
            h = 0.0001
            #minmaxvar
            sensitivity_lambda_bidminmaxvar =  (minmaxvar_bid_model(payoff, mmv_minlambda + h) - minmaxvar_bid_model(payoff, mmv_minlambda - h))/2*h
            sensitivity_lambda_askminmaxvar =  (minmaxvar_ask_model(payoff, mmv_minlambda + h) - minmaxvar_ask_model(payoff, mmv_minlambda - h))/2*h
            #wang
            sensitivity_lambda_bidwang =  (Wang_bid_model(payoff, mmv_minlambda + h) - Wang_bid_model(payoff, mmv_minlambda - h))/2*h
            sensitivity_lambda_askwang =  (Wang_ask_model(payoff, mmv_minlambda + h) - Wang_ask_model(payoff, mmv_minlambda - h))/2*h
            #t_minmaxwang
            sensitivity_lambda_bidtminmaxwang =  (t_minmax_wang_bid_model(payoff, mmv_minlambda + h, t) - t_minmax_wang_bid_model(payoff, mmv_minlambda - h, t))/2*h
            sensitivity_lambda_asktminmaxwang =  (t_minmax_wang_ask_model(payoff, mmv_minlambda + h, t) - t_minmax_wang_ask_model(payoff, mmv_minlambda - h, t))/2*h
            #Comp Wang MinmaxVar
            sensitivity_lambda_bidcompwangminmax =  (comp_wang_minmaxvar_bid_model(payoff, mmv_minlambda + h) -  comp_wang_minmaxvar_bid_model(payoff, mmv_minlambda - h))/2*h
            sensitivity_lambda_askcompwangminmax =  (comp_wang_minmaxvar_ask_model(payoff, mmv_minlambda + h) -  comp_wang_minmaxvar_ask_model(payoff, mmv_minlambda - h))/2*h
            #Comp Minmaxvar Wang
            sensitivity_lambda_bidcompminmaxwang =  ( comp_minmaxvar_wang_bid_model(payoff, mmv_minlambda + h) -   comp_minmaxvar_wang_bid_model(payoff, mmv_minlambda - h))/2*h
            sensitivity_lambda_askcompminmaxwang =  ( comp_minmaxvar_wang_ask_model(payoff, mmv_minlambda + h) -   comp_minmaxvar_wang_ask_model(payoff, mmv_minlambda - h))/2*h            

            #Store them them for K_i
            # #MinMaxVar
            # minmaxvar_impliedliquiditysurface_K.append(mmv_minlambda)
            # minmaxvar_bidsmodel_K.append(mmv_bidmodel)
            # minmaxvar_asksmodel_K.append(mmv_askmodel)
            # #Wang
            # Wang_impliedliquiditysurface_K.append(wang_minlambda)
            # Wang_bidsmodel_K.append(wang_bidmodel)
            # Wang_asksmodel_K.append(wang_askmodel)
            #Store them them for K_i
            #Minmaxvar
            minmaxvar_impliedliquiditysurface_K.append(mmv_minlambda)
            minmaxvar_bidsmodel_K.append(mmv_bidmodel)
            sensitivity_minmaxvar_bidsmodel_K.append(sensitivity_lambda_bidminmaxvar)
            minmaxvar_asksmodel_K.append(mmv_askmodel)
            sensitivity_minmaxvar_asksmodel_K.append(sensitivity_lambda_askminmaxvar)
            #Wang
            Wang_impliedliquiditysurface_K.append(wang_minlambda)
            Wang_bidsmodel_K.append(wang_bidmodel)
            sensitivity_Wang_bidsmodel_K.append(sensitivity_lambda_bidwang)
            Wang_asksmodel_K.append(wang_askmodel)
            sensitivity_Wang_asksmodel_K.append(sensitivity_lambda_askwang)
            #t-MinMax-Wang
            t_minmaxWang_impliedliquiditysurface_K.append(mnv_wang_minlambda)
            t_minmaxWang_bidsmodel_K.append(mnv_wang_bidmodel)
            sensitivity_t_minmaxWang_bidsmodel_K.append(sensitivity_lambda_bidtminmaxwang )
            t_minmaxWang_asksmodel_K.append(mnv_wang_askmodel)
            sensitivity_t_minmaxWang_asksmodel_K.append(sensitivity_lambda_asktminmaxwang)
            t_minmaxWang_K.append(t)
            #Comp WangMinmax
            comp_Wangminmaxvar_impliedliquiditysurface_K.append(comp_wang_minmaxvar_minlambda)
            comp_Wangminmax_bidsmodel_K.append(comp_wang_minmaxvar_bidmodel)
            sensitivity_comp_Wangminmax_bidsmodel_K.append(sensitivity_lambda_bidcompwangminmax )
            comp_Wangminmax_asksmodel_K.append(comp_wang_minmaxvar_askmodel)
            sensitivity_comp_Wangminmax_asksmodel_K.append(sensitivity_lambda_askcompwangminmax )
            #Comp MinmmaxWang
            comp_minmaxWang_impliedliquiditysurface_K.append(comp_minmaxvar_wang_minlambda)
            comp_minmaxWang_bidsmodel_K.append(comp_minmaxvar_wang_bidmodel)
            sensitivity_comp_minmaxWang_bidsmodel_K.append(sensitivity_lambda_bidcompminmaxwang)
            comp_minmaxWang_asksmodel_K.append(comp_minmaxvar_wang_askmodel)
            sensitivity_comp_minmaxWang_asksmodel_K.append(sensitivity_lambda_askcompminmaxwang)
            #
        #Store them in the list for maturity M_i
        #minMaxvar
        minmaxvar_impliedliquiditysurface_MK.append(minmaxvar_impliedliquiditysurface_K)
        minmaxvar_bidsmodel_MK.append(minmaxvar_bidsmodel_K)
        sensitivity_minmaxvar_bidsmodel_MK.append(sensitivity_minmaxvar_bidsmodel_K)
        minmaxvar_asksmodel_MK.append(minmaxvar_asksmodel_K)
        sensitivity_minmaxvar_asksmodel_MK.append(sensitivity_minmaxvar_asksmodel_K)
        #Wang
        Wang_impliedliquiditysurface_MK.append(Wang_impliedliquiditysurface_K)
        Wang_bidsmodel_MK.append(Wang_bidsmodel_K)
        sensitivity_Wang_bidsmodel_MK.append(sensitivity_Wang_bidsmodel_K)
        Wang_asksmodel_MK.append(Wang_asksmodel_K)
        sensitivity_Wang_asksmodel_MK.append(sensitivity_Wang_asksmodel_K)
        #t-MinMaxWang
        t_minmaxWang_impliedliquiditysurface_MK.append(t_minmaxWang_impliedliquiditysurface_K)
        t_minmaxWang_bidsmodel_MK.append(t_minmaxWang_bidsmodel_K)
        sensitivity_t_minmaxWang_bidsmodel_MK.append(sensitivity_t_minmaxWang_bidsmodel_K)
        t_minmaxWang_asksmodel_MK.append(t_minmaxWang_asksmodel_K)
        sensitivity_t_minmaxWang_asksmodel_MK.append(sensitivity_t_minmaxWang_asksmodel_K)
        t_minmaxWang_MK.append(t_minmaxWang_K)
        #CompWangMinMax
        comp_Wangminmaxvar_impliedliquiditysurface_MK.append(comp_Wangminmaxvar_impliedliquiditysurface_K)
        comp_Wangminmax_bidsmodel_MK.append(comp_Wangminmax_bidsmodel_K)
        sensitivity_comp_Wangminmax_bidsmodel_MK.append(sensitivity_comp_Wangminmax_bidsmodel_K)
        comp_Wangminmax_asksmodel_MK.append(comp_Wangminmax_asksmodel_K)
        sensitivity_comp_Wangminmax_asksmodel_MK.append(sensitivity_comp_Wangminmax_asksmodel_K)
        #CompMinMaxWang
        comp_minmaxWang_impliedliquiditysurface_MK.append(comp_minmaxWang_impliedliquiditysurface_K)
        comp_minmaxWang_bidsmodel_MK.append(comp_minmaxWang_bidsmodel_K)
        sensitivity_comp_minmaxWang_bidsmodel_MK.append(sensitivity_comp_minmaxWang_bidsmodel_K)
        comp_minmaxWang_asksmodel_MK.append(comp_minmaxWang_asksmodel_K)
        sensitivity_comp_minmaxWang_asksmodel_MK.append(sensitivity_comp_minmaxWang_asksmodel_K)
    return minmaxvar_impliedliquiditysurface_MK, minmaxvar_bidsmodel_MK, sensitivity_minmaxvar_bidsmodel_MK, minmaxvar_asksmodel_MK, sensitivity_minmaxvar_asksmodel_MK, Wang_impliedliquiditysurface_MK, Wang_bidsmodel_MK, sensitivity_Wang_bidsmodel_MK, Wang_asksmodel_MK, sensitivity_Wang_asksmodel_MK, t_minmaxWang_impliedliquiditysurface_MK, t_minmaxWang_bidsmodel_MK, sensitivity_t_minmaxWang_bidsmodel_MK, t_minmaxWang_asksmodel_MK, sensitivity_t_minmaxWang_asksmodel_MK, t_minmaxWang_MK, comp_Wangminmaxvar_impliedliquiditysurface_MK, comp_Wangminmax_bidsmodel_MK, sensitivity_comp_Wangminmax_bidsmodel_MK, comp_Wangminmax_asksmodel_MK, sensitivity_comp_Wangminmax_asksmodel_MK, comp_minmaxWang_impliedliquiditysurface_MK, comp_minmaxWang_bidsmodel_MK, sensitivity_comp_minmaxWang_bidsmodel_MK, comp_minmaxWang_asksmodel_MK, sensitivity_comp_minmaxWang_asksmodel_MK

# #Lets Run the functions
# payoffs_MK, mean_payoffs_MK = get_mc_payoffs(real_data, data)
payoffs_MK = data['mc_prices_model'] 

#
def get_mean_payoffs(payoffs_MK):
    K = get_strikes(real_data)
    mean_payoffs_MK = []
    for i in range(len(data['mc_prices_model'])):
            mean_payoffs_K = []
            for k in range(len(K[0])):
                payoff = payoffs_MK[i][k]
                mean_payoff = np.mean(payoff)
                #Append payoff for strike K_i 
                mean_payoffs_K.append(mean_payoff)
            #Append all payoffs for maturity M_i  
            mean_payoffs_MK.append(mean_payoffs_K)
    return mean_payoffs_MK


mean_payoffs_MK = get_mean_payoffs(payoffs_MK)

print("THESE ARE THE MEAN PAYOFFS OF THE MC PATHS")
print(mean_payoffs_MK)

minmaxvar_impliedliquiditysurface_MK, minmaxvar_bidsmodel_MK, sensitivity_minmaxvar_bidsmodel_MK, minmaxvar_asksmodel_MK, sensitivity_minmaxvar_asksmodel_MK, Wang_impliedliquiditysurface_MK, Wang_bidsmodel_MK, sensitivity_Wang_bidsmodel_MK, Wang_asksmodel_MK, sensitivity_Wang_asksmodel_MK, t_minmaxWang_impliedliquiditysurface_MK, t_minmaxWang_bidsmodel_MK, sensitivity_t_minmaxWang_bidsmodel_MK, t_minmaxWang_asksmodel_MK, sensitivity_t_minmaxWang_asksmodel_MK, t_minmaxWang_MK, comp_Wangminmaxvar_impliedliquiditysurface_MK, comp_Wangminmax_bidsmodel_MK, sensitivity_comp_Wangminmax_bidsmodel_MK, comp_Wangminmax_asksmodel_MK, sensitivity_comp_Wangminmax_asksmodel_MK, comp_minmaxWang_impliedliquiditysurface_MK, comp_minmaxWang_bidsmodel_MK, sensitivity_comp_minmaxWang_bidsmodel_MK, comp_minmaxWang_asksmodel_MK, sensitivity_comp_minmaxWang_asksmodel_MK = bid_ask_cali(real_data, payoffs_MK)

print("THESE ARE THE BIDS AND ASKS AFTER CALIBRATION")
#Save the data in the calibartion file 
np.savez(parent_dir + '\\PastCalibrations\\'  + calibration_name + '\\'+'bidaskCaliData',  mean_payoffs_MK = mean_payoffs_MK, 
             minmaxvar_impliedliquiditysurface_MK = minmaxvar_impliedliquiditysurface_MK, minmaxvar_bidsmodel_MK = minmaxvar_bidsmodel_MK,
             sensitivity_minmaxvar_bidsmodel_MK = sensitivity_minmaxvar_bidsmodel_MK, minmaxvar_asksmodel_MK = minmaxvar_asksmodel_MK, sensitivity_minmaxvar_asksmodel_MK = sensitivity_minmaxvar_asksmodel_MK, Wang_impliedliquiditysurface_MK =  Wang_impliedliquiditysurface_MK,
             Wang_bidsmodel_MK = Wang_bidsmodel_MK, sensitivity_Wang_bidsmodel_MK = sensitivity_Wang_bidsmodel_MK, Wang_asksmodel_MK = Wang_asksmodel_MK, sensitivity_Wang_asksmodel_MK = sensitivity_Wang_asksmodel_MK, t_minmaxWang_impliedliquiditysurface_MK =  t_minmaxWang_impliedliquiditysurface_MK, 
             t_minmaxWang_bidsmodel_MK = t_minmaxWang_bidsmodel_MK , sensitivity_t_minmaxWang_bidsmodel_MK=sensitivity_t_minmaxWang_bidsmodel_MK, t_minmaxWang_asksmodel_MK = t_minmaxWang_asksmodel_MK, 
             sensitivity_t_minmaxWang_asksmodel_MK = sensitivity_t_minmaxWang_asksmodel_MK, t_minmaxWang_MK = t_minmaxWang_MK, comp_Wangminmaxvar_impliedliquiditysurface_MK  = comp_Wangminmaxvar_impliedliquiditysurface_MK,
             comp_Wangminmax_bidsmodel_MK  = comp_Wangminmax_bidsmodel_MK, sensitivity_comp_Wangminmax_bidsmodel_MK = sensitivity_comp_Wangminmax_bidsmodel_MK,
             comp_Wangminmax_asksmodel_MK=comp_Wangminmax_asksmodel_MK, sensitivity_comp_Wangminmax_asksmodel_MK=sensitivity_comp_Wangminmax_asksmodel_MK, comp_minmaxWang_impliedliquiditysurface_MK =  comp_minmaxWang_impliedliquiditysurface_MK,
             comp_minmaxWang_bidsmodel_MK = comp_minmaxWang_bidsmodel_MK, sensitivity_comp_minmaxWang_bidsmodel_MK=sensitivity_comp_minmaxWang_bidsmodel_MK,
             comp_minmaxWang_asksmodel_MK = comp_minmaxWang_asksmodel_MK, sensitivity_comp_minmaxWang_asksmodel_MK =  sensitivity_comp_minmaxWang_asksmodel_MK)