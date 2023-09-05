'''
download kauai dataset for all 200 Hz sensors in OOI
'''

import kaooi
import xarray as xr
import multiprocessing as mp
from tqdm import tqdm
import os
import kaooi



if __name__ == '__main__':
    # Get the number of CPUs available
    num_cpus = mp.cpu_count()
    # Create a pool of processes, one for each CPU
    
    Tx_times = kaooi.get_Tx_keytimes(year=2023)
    with mp.Pool(processes=num_cpus) as pool:
        list(tqdm(pool.imap_unordered(kaooi.downloadTx_200hz, Tx_times, chunksize=1), total=len(Tx_times)))