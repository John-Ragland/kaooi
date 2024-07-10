'''
download kauai dataset for all 200 Hz sensors in OOI
'''

import kaooi
import xarray as xr
import multiprocessing as mp
from tqdm import tqdm
import os
import kaooi
import pandas as pd

Tx_time = pd.Timestamp('2023-03-18T02:00:00')
kaooi.downloadTx_200hz(Tx_time, ds_dir='/datadrive/kauai/transmissions/200Hz_sensors_clipped_noTx/', verbose=True)
