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
    # Get the number of CPUs available
    num_cpus = mp.cpu_count()
    # Create a pool of processes, one for each CPU

    Tx_times = kaooi.get_Tx_keytimes(year=2023)[:119]

    # create partial function
    download_Tx_partial = functools.partial(
        kaooi.downloadTx_64kHz,
        ds_dir='/datadrive/kauai/transmissions/ooi_bb/',
        length='2H',
        verbose=True,
        preprocess=False,
    )

    for Tx_time in tqdm(Tx_times):

        # Throw out any data for when ooi broke the hydrophones ._.
        if Tx_time > pd.Timestamp('2023-06-07'):
            continue

        download_Tx_partial(Tx_time)
