'''
download kauai dataset for all 200 Hz sensors in OOI
'''

import kaooi
import xarray as xr
import multiprocessing as mp
from tqdm import tqdm
import os
import kaooi
import functools
import pandas as pd

if __name__ == '__main__':

    Tx_times = kaooi.get_Tx_keytimes(year=2023)
    
    for Tx_time in tqdm(Tx_times):
        kaooi.downloadTx_64kHz(
            Tx_time,
            ds_dir='/datadrive/kauai/transmissions/ooi_bb/LJ01C',
            length='2H',
            verbose=True,
            preprocess=False,
            nodes=['LJ01C']
        )