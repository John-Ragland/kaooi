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

if __name__ == '__main__':
    # Get the number of CPUs available
    num_cpus = mp.cpu_count()

    for year in [2023, 2024]:
        Tx_times = kaooi.get_Tx_keytimes(year=year)

        # create partial function
        download_Tx_partial = functools.partial(
            kaooi.downloadTx_200hz,
            ds_dir='/datadrive/kauai/transmissions/ooi_lf/',
            length='2h',
            verbose=True,
        )

        # Create a pool of processes, one for each CPU

        with mp.Pool(processes=num_cpus) as pool:
            list(tqdm(pool.imap_unordered(download_Tx_partial, Tx_times, chunksize=1), total=len(Tx_times)))
