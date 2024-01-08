'''
signal processing
'''

import kaooi
import xarray as xr
import xrsignal as xrs
import numpy as np
import pandas as pd

def match_filter(
        ds: xr.Dataset,
        dim: str,
        sampling_rate: float,
        length: str = '2H',
) -> xr.Dataset:
    '''
    take xarray dataset and apply match filter to each channel

    Parameters
    ----------
    ds : xr.Dataset
        dataset of hydrophone /seismometer data
    dim : str
        dimension to match filter along
    sampling_rate : float
        sampling rate of data
    length : str
        length of time segment, should be readable by pd.Timedelta
    
    Returns
    -------
    ds_match : xr.Dataset
        match filtered dataset of hydrophone /seismometer data
        will have complex values
    '''
    replica = kaooi.construct_replica(sampling_rate = sampling_rate)

    # keep only integer number of replicas in ds
    ds = ds.isel({dim:slice(0, int(ds[dim].size/replica[dim].size)*replica[dim].size)})

    ds_match = xrs.correlate(ds, replica, mode='same')
    
    return ds_match
    # construct new time coordinates
    len_rep = len(replica)/sampling_rate
    lt = np.arange(0,pd.Timedelta(length).value/1e9,len_rep)[:-1]
    st = np.arange(0,len_rep, 1/sampling_rate)
    ltlt, stst = np.meshgrid(lt, st)

    stst = stst.T.flatten()
    ltlt = ltlt.T.flatten()

    ds_match = ds_match.assign_coords({'longtime': (dim, ltlt), 'shorttime': (dim, stst)})

    return ds_match