'''
update_dataset.py - this script saves each transmission as a seperate netCDF file
'''

import kaooi
import xarray
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import zarr
from tqdm import tqdm

# get all transmissions up to present
Tx_times = kaooi.get_Tx_keytimes(2023)[40:]
length = '2H'

for Tx_time in tqdm(Tx_times):
    # Download data from ooi
    fn = f'/datadrive/kauai/transmissions/{Tx_time.strftime("%Y%m%dT%H%M%S")}.nc'
    
    # skip Tx time if file already exists
    if os.path.exists(fn):
        print(f'{Tx_time} already exists')
        continue
    else:
        hdata = kaooi.get_ooi_data(Tx_time, length=length, verbose=False)
        if hdata != None:
            hdata.to_netcdf(fn)