import numpy as np
import pandas as pd
import openpyxl
import numpy as np
import os

from requests import get

# get the parent path
parent_dir = os.path.dirname(os.path.abspath(__file__))

def get_spot(File_name):
    df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Underlying_spot")
    numpy_array = df.to_numpy()
    spot = numpy_array[0][1]
    return(float(spot))


def get_maturities(File_name):
    df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Maturities")
    df = df["Maturities as year fractions"]
    maturities = df.values.tolist()
    maturities = [float(i) for i in maturities]
    return(maturities)


def get_strikes(File_name):
    df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Strikes")
    df = df.set_index("Index")
    strikes = df.values.tolist()
    strikes = np.float_(strikes)
    strikes = np.array(strikes)
    strikes = list(strikes)
    strikes = [np.array([x for x in a if not np.isnan(x)]) for a in strikes]
    return(strikes)

def get_mid_prices(File_name):
    df1 = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Bids")
    df1 = df1.set_index("Index")
    df2 = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Asks")
    df2 = df2.set_index("Index")
    mid = pd.concat([df1, df2]).groupby(level=0).mean()
    prices = mid.values.tolist()
    prices = np.float_(prices)
    prices = np.array(prices)
    prices = list(prices)
    return(prices)

# def get_mid_prices(File_name):
#     df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Risk_Neutral")
#     df = df.set_index("Index")
#     rn = df.values.tolist()
#     rn = np.float_(rn)
#     rn = np.array(rn)
#     rn = [np.array([x for x in a if not np.isnan(x)]) for a in rn]
#     return(rn)

def get_bids(File_name):
    df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Bids")
    df = df.set_index("Index")
    bids = df.values.tolist()
    bids = np.float_(bids)
    bids = np.array(bids)
    bids = [np.array([x for x in a if not np.isnan(x)]) for a in bids]
    return(bids)

def get_asks(File_name):
    df = pd.read_excel( parent_dir + "\\real_data\\" + File_name + ".xlsx", sheet_name="Asks")
    df = df.set_index("Index")
    asks = df.values.tolist()
    asks = np.float_(asks)
    asks = np.array(asks)
    asks = list(asks)
    asks = [np.array([x for x in a if not np.isnan(x)]) for a in asks]
    return(asks)



print(get_maturities('BTC_surface'))
