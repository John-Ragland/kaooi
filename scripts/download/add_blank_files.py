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
    nodes = ['LJ01C', 'PC01A', 'PC03A', 'LJ01A', 'LJ01D']
    for node in nodes:
        print(f'filling in blank files for {node}...')
        ds_dir = f'/gscratch/coenv/jhrag/data/kauai/transmissions/ooi_bb_tc/{node}'
        for year in [2023, 2024]:

            # most recent update of dataset
            last_update = pd.Timestamp('2024-07-16 T12:00:00')
            
            Tx_times = kaooi.get_Tx_keytimes(year=year)

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
                    hdata = {node: None}
                    print(node, Tx_time)
                    ds_blank = kaooi.construct_xds(Tx_time, hdata, length='2h', sampling_rate=500)
                    # save to disk
                    ds_blank.to_netcdf(fn)
