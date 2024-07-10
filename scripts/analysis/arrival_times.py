'''
# arrival_times.py
*compute the arrival times for all transmissions*

- dataset openned with ::kaooi.open_ooi_bb():: and ::kaooi.open_ooi_lf()::
- processing computed with ::kaooi.process_data()::

## arrival time data structure
- Node (i.e. LJ01C)
    - peak_locs # list of peak locations (length N) 
        - [ ... ] # np.array (arbitrary length) of peak locations for specific transmit time in seconds (circular time)
        - [ ... ]
            .
            .
            .
    - snr
        - [ ... ] # np.array (arbitrary length) of snr for specific transmit time
        - [ ... ]
            .
            .
            .
    - transmit_time
        - [ ... ] # np.array (shape (N,)), of transmit times
            

'''


import kaooi
from dask.distributed import LocalCluster, Client
import dask
import xarray as xr
import pandas as pd
import numpy as np
import json

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    bb_nodes = ['LJ01C','PC01A','PC03A']

    # manually chosen to be outside of arrival range
    noise_slices_bb = {
        'LJ01C': slice(20, 27.28),
        'PC01A': slice(0, 7.28),
        'PC03A': slice(20, 27.28),
    }

    ds_bb = kaooi.open_ooi_bb()
    ds_bb_proc = kaooi.process_data(ds_bb, sampling_rate=500)
    ds_bb_stack = np.abs(ds_bb_proc).sel({'longtime': slice(40*60, 65*60)}).mean('longtime')

    arrival_times = {}

    for node in bb_nodes:
        arrival_times[node] = kaooi.estimate_peaks_array(
            ds_bb_stack[node].compute(),
            noise_slice=noise_slices_bb[node]
        )

    # write to json file
    with open("/datadrive/kauai/analysis/arrival_times_bb.json", "w") as json_file:
        json.dump(arrival_times, json_file)
