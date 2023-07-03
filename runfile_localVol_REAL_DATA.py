#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
# Copyright (c) 2020 Christa Cuchiero, Wahid Khosrawi, Josef Teichmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""runfile_localVol.py:
This file implements the the functionality for computing and plotting local-vol 
implied volatilities for the sampled parameters given as input and options also
given as input. Starting point is the main function.
"""


import matplotlib.pyplot as plt
import numpy as np, tensorflow as tf
import os
import platform

from compute_pM import MC_pM_locvol  #LINE 174
from finModels.helpers_finModels import fin_surface

import pickle
import helpers

# Env variables
HOSTNAME = platform.uname()[1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Disable_eager_execution
tf.compat.v1.disable_eager_execution()

# get the parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))

#Name of xlsx Real Data file you want to calibrate
data = "Cali_6mat20k" 

#Import Fuunction to get mid-prices

from get_data import get_mid_prices

def makePrices(para_locVol, option, para_MC, just_ops = False):
    '''Compute locVol prices for the given parameters. '''
    version = para_locVol['version']
    path = os.path.join(parent_dir, 'caliRes\\'+version)
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    
    pM = get_mid_prices(data)

    #Printing the option prices shape and the option prices
    print("pM Shape and pM are:")
    print(np.shape(pM))
    print(pM)

    # Save the prices to disk for later calibration
    with open(parent_dir + '\\data\\locVolPrices\\pM_'+version+'.pkl', 'wb') as f:
       pickle.dump(pM, f) 

def plot_IV(para, option):
    '''load computed prices and create fin_surf instance. Also do all needed conversions'''

    log_m = option.log_m_list(para['S0'])
    version = para['version']

    with open(parent_dir + '\\data\\locVolPrices\\pM_'+version+'.pkl', 'rb') as f:
       pM = pickle.load(f)

    #Print what is variable option 
    print("The variable option is:")
    print(option)



    # Constuct the object that handles conversion
    data_handler = fin_surface(mats=option.T, strikes=option.K, spot=para['S0'], version = version)
    data_handler.paralocVol = para
    data_handler.feed_prices(prices= [p for p in pM]  )    
    data_handler.convert(direction='price->iV')

    for i in range(len(pM)):
        plt.plot(log_m[i], data_handler.iV[i] )
        if not os.path.exists(parent_dir + '\\caliRes\\'+version):
            os.mkdir(parent_dir + '\\caliRes\\'+version)
        plt.savefig(parent_dir + '\\caliRes\\'+version+'\\plot_{}.png'.format(str(i+1).zfill(3)) )
        plt.close()

    return(data_handler)


def main(para_locVol, option, para_MC, compute_prices):

    '''wrapper function. Depending on compute_prices data is computed or just loaded.'''
    if compute_prices:
        makePrices(para_locVol, option, para_MC)

    
    finsurf = plot_IV(para_locVol, option)
    return(finsurf)


if __name__ == "__main__":
    pass