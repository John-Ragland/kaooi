'''
initialize_dataset.py - this scripts creates a zarr file for the first kauai source broadcast in 2023.
This script must be run to create the dataset before you can run update_dataset.py to include data
from new broadcasts.
'''

import kaooi
import xarray
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import zarr

# first transmission of 2023
Tx_time = kaooi.get_Tx_keytimes(2023)[0]
length = '2H'

# Download data from ooi
hdata = kaooi.get_ooi_data(Tx_time, length=length, verbose=True)

print('writing to zarr store (Azure Blob Storage)')
# save to Azure Blob Storage
connection_string = os.environ['AZURE_CONSTR_ooidata']
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "kauai"
container_client = blob_service_client.get_container_client(container_name)

# Create the Zarr store using ABSStore
output_path = 'kauai_transmissions_ooi.zarr'

zarr_store = zarr.storage.ABSStore(
    client = container_client,
    prefix=output_path,
)

hdata.to_zarr(store=zarr_store, mode="w-")
print('complete.')