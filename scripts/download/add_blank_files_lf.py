'''
for transmission times, where there is uneven data coverage and the file was not saved to disk,
    add a netcdf file filled with nan values so that the dataset can be opened.
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


    ds_dir = f'/datadrive/kauai/transmissions/ooi_lf'
    for year in [2023, 2024]:

        # most recent update of dataset
        # last_update = pd.Timestamp('today')
        last_update = pd.Timestamp('2024-02-11')
        Tx_times = kaooi.get_Tx_keytimes()

        for Tx_time in Tx_times:
            # check if time is past last update
            if Tx_time > last_update:
                continue
            # check if data already exists
            fn = f'{ds_dir}/{Tx_time.strftime("%Y%m%dT%H%M%S")}.nc'

            if os.path.exists(fn):
                continue
            else:
                print(f'\t{Tx_time}')
                # create empty dataset
                hdata = {
                    'AXBA1': None,
                    'AXCC1': None,
                    'AXEC2': None, 
                    'HYS14': None,
                    'HYSB1': None
                }

                ds_blank = kaooi.construct_xds(Tx_time, hdata, length='2h', sampling_rate=200)
                # save to disk
                ds_blank.to_netcdf(fn)
                print(fn, 'added')
