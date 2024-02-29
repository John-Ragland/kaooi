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
    carrier_freq = 75
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
    carrier_freq : Optional[float]
        carrier frequency of match filter, defaul is 75 Hz

    Returns
    -------
    ds_match : xr.Dataset
        match filtered dataset of hydrophone /seismometer data
        will have complex values
    '''
    replica = kaooi.construct_replica(sampling_rate=sampling_rate, carrier_freq=carrier_freq)

    # keep only integer number of replicas in ds
    #ds = ds.isel({dim: slice(0, int(ds[dim].size / replica[dim].size) * replica[dim].size)})

    ds_match = xrs.correlate(ds, replica, mode='same')

    return ds_match
    # construct new time coordinates
    len_rep = len(replica) / sampling_rate
    lt = np.arange(0, pd.Timedelta(length).value / 1e9, len_rep)[:-1]
    st = np.arange(0, len_rep, 1 / sampling_rate)
    ltlt, stst = np.meshgrid(lt, st)

    stst = stst.T.flatten()
    ltlt = ltlt.T.flatten()

    ds_match = ds_match.assign_coords({'longtime': (dim, ltlt), 'shorttime': (dim, stst)})

    return ds_match


def process_data(ds: xr.Dataset, sampling_rate: float = 500, carrier_freq = 75) -> xr.Dataset:
    '''
    process_data - compute pre-processing and match filtering to raw data
    - clip data to $\pm 4$ stds
    - match filter
    - compute hilbert transform of data
    - restructure data to longtime / shorttime

    Parameters
    ----------
    ds : xr.Dataset
        dataset of hydrophone /seismometer data
    sampling_rate : float
        sampling rate of data
    carrier_freq : Optional[float]
        carrier frequency of match filter, default is 75 Hz

    Returns
    -------
    ds_processed : xr.Dataset
        dataset with pre-processed and match filtered data
    '''

    ds_std = ds.std('time')
    ds_clip = ds.clip(ds_std * -4, ds_std * 4)
    ds_match = kaooi.match_filter(ds_clip, dim='time', sampling_rate=sampling_rate, length='2h', carrier_freq=carrier_freq)
    ds_c = xrs.hilbert(ds_match, dim='time')
    ds_rs = kaooi.add_ltst_coords(ds_c, dim='time', sampling_rate=sampling_rate, length='2h', carrier_freq=carrier_freq)
    ds_processed = ds_rs.set_index({'time': ('longtime', 'shorttime')}).unstack()
    
    return ds_processed
