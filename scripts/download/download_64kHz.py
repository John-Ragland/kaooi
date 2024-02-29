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

    nodes = ['LJ01C', 'PC01A', 'PC03A']#, 'LJ01A', 'LJ01D']
    for node in nodes:
        print(f'downloading data for {node}...')
        for year in [2023, 2024]:
            Tx_times = kaooi.get_Tx_keytimes(year=year)
            stop_at = pd.Timestamp('today')
            for Tx_time in tqdm(Tx_times):
                if Tx_time > stop_at:
                    continue
                kaooi.downloadTx_64kHz(
                    Tx_time,
                    ds_dir=f'/datadrive/kauai/transmissions/ooi_bb_tc/{node}',
                    length='2h',
                    verbose=True,
                    preprocess=False,
                    nodes=[node]
                )