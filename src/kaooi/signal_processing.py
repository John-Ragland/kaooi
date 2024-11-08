'''
signal processing
'''

import kaooi
import xarray as xr
import xrsignal as xrs
import numpy as np
import pandas as pd
import scipy
from scipy import signal
from typing import Union

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


def process_data(
        ds: xr.Dataset,
        sampling_rate: float = 500,
        carrier_freq = 75
    ) -> xr.Dataset:
    '''
    process_data - compute pre-processing and match filtering to raw data
    - clip data to +- 4 stds
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


def process_data_filt(
        ds: xr.Dataset,
        sampling_rate: float = 500,
        carrier_freq = 75
    ) -> xr.Dataset:
    '''
    process_data - compute pre-processing and match filtering to raw data
    - clip data to +- 4 stds
    - filter data between 37.5 and 112.5 Hz
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

    if sampling_rate > 300:
        b,a = signal.butter(4, [37.5 / (sampling_rate / 2), 112.5 / (sampling_rate / 2)], btype='bandpass')
    else:
        b,a = signal.butter(4, 37.5 / (sampling_rate / 2), btype='highpass')

    ds_filt = xrs.filtfilt(ds, b=b, a=a, dim='time')
    ds_std = ds_filt.std('time')
    ds_clip = ds.clip(ds_std * -4, ds_std * 4)
    ds_match = kaooi.match_filter(ds_clip, dim='time', sampling_rate=sampling_rate, length='2h', carrier_freq=carrier_freq)
    ds_c = xrs.hilbert(ds_match, dim='time')
    ds_rs = kaooi.add_ltst_coords(ds_c, dim='time', sampling_rate=sampling_rate, length='2h', carrier_freq=carrier_freq)
    ds_processed = ds_rs.set_index({'time': ('longtime', 'shorttime')}).unstack()
    
    return ds_processed

def process_fwhiten(
    ds : xr.Dataset,
    T0s : dict,
    sampling_rate: float = 500,
    carrier_freq = 75
) -> xr.Dataset:
    '''
    process_fwhiten - process the data with the method of frequency whitening.
    Only consider the twenty minutes of the reception of the signal, which is
    estimated from the mode 1 group velocites provided by T0s, (or more conveniently,
    the average sound speed minimum along the path)

    **processeing steps**
    - clip data to +- 4 stds
    - frequency whiten 20 minute segments of data
    - match filter
    - compute hilbert transform of data
    - restructure data to longtime / shorttime

    Parameters
    ----------
    ds : xr.Dataset
        dataset of hydrophone data. Should have dimensions ['time', 'transmission']
    T0s : dict
        dictionary of mode 1 group velocities, or average sound speed minimum along the path
    sampling_rate : float
        sampling rate of data
    carrier_freq : Optional[float]
        carrier frequency of match filter, default is 75 Hz

    Returns
    -------
    ds_processed : xr.Dataset
        dataset with pre-processed and match filtered data. Will have 
        dimensions ['longtime', 'shorttime', 'transmission']
    '''

    # check that keys in T0 match keys in ds
    assert ds.keys() == T0s.keys(), f'keys in T0 {T0s.keys()} much match keys in ds {list(ds.keys())}'
    
    # slice data to 20 minutes around T0
    sliced_data = {}
    for node in ds.keys():
        sliced_data[node] = ds[node].sel(time=slice(T0s[node], T0s[node] + (20*60))).drop_vars('time')

    ds_sliced = xr.Dataset(sliced_data).assign_coords({'time': np.arange(0, 20*60, 1/sampling_rate)})


    ds_std = ds_sliced.std('time')
    ds_clip = ds_sliced.clip(ds_std * -4, ds_std * 4)

    # whiten depending on the sampling rate
    if sampling_rate == 500:
        b,a = signal.butter(4, [30 / (500 / 2), 120 / (500 / 2)], btype='band')
    elif sampling_rate == 200:
        b,a = signal.butter(4, [30 / (200 / 2), 90 / (200 / 2)], btype='band')
    ds_white = freq_whiten(ds_clip, b=b, a=a, dim='time')

    ds_match = kaooi.match_filter(ds_white, dim='time', sampling_rate=sampling_rate, length='20m', carrier_freq=carrier_freq)
    ds_c = xrs.hilbert(ds_match, dim='time')
    ds_rs = kaooi.add_ltst_coords(ds_c, dim='time', sampling_rate=sampling_rate, length='20m', carrier_freq=carrier_freq)
    ds_processed = ds_rs.set_index({'time': ('longtime', 'shorttime')}).unstack()
    
    return ds_processed

def process_fwhiten_27(
    ds : xr.Dataset,
    T0s : dict,
    sampling_rate: float = 500,
    carrier_freq = 75
) -> xr.Dataset:
    '''
    process_fwhiten - process the data with the method of frequency whitening.
    Only consider the twenty minutes of the reception of the signal, which is
    estimated from the mode 1 group velocites provided by T0s, (or more conveniently,
    the average sound speed minimum along the path)
    sequence of processing is:
    - slice data between T0 and T0 + 20min
    - clip data to +- 4 stds
    - frequency whiten each 27.28 segment of data
    - match filter

    Parameters
    ----------
    ds : xr.Dataset
        dataset of hydrophone data. Should have dimensions ['time', 'transmission']
    T0s : dict
        dictionary of mode 1 group velocities, or average sound speed minimum along the path
    sampling_rate : float
        sampling rate of data
    carrier_freq : Optional[float]
        carrier frequency of match filter, default is 75 Hz

    Returns
    -------
    ds_processed : xr.Dataset
        dataset with pre-processed and match filtered data. Will have 
        dimensions ['longtime', 'shorttime', 'transmission']
    '''
    # check that keys in T0 match keys in ds
    assert ds.keys() == T0s.keys(), f'keys in T0 {T0s.keys()} much match keys in ds {list(ds.keys())}'
    
    # slice data to 20 minutes around T0
    sliced_data = {}
    for node in ds.keys():
        sliced_data[node] = ds[node].sel(time=slice(T0s[node], T0s[node] + (20*60))).drop_vars('time')

    ds_sliced = xr.Dataset(sliced_data).assign_coords({'time': np.arange(0, 20*60, 1/sampling_rate)})


    ds_std = ds_sliced.std('time')
    ds_clip = ds_sliced.clip(ds_std * -4, ds_std * 4)

    # reshape data
    ds_lsts = kaooi.add_ltst_coords(ds_clip, dim='time', sampling_rate=sampling_rate, length='20m')
    ds_rc = ds_lsts.set_index({'time': ('longtime', 'shorttime')}).unstack()
    
    # whiten depending on the sampling rate
    if sampling_rate == 500:
        b,a = signal.butter(4, [30 / (500 / 2), 120 / (500 / 2)], btype='band')
    elif sampling_rate == 200:
        b,a = signal.butter(4, [30 / (200 / 2), 90 / (200 / 2)], btype='band')
    ds_whiten = freq_whiten(ds_rc, b=b, a=a, dim='shorttime')
    ds_whiten = ds_whiten.stack({'time':('longtime', 'shorttime')}, create_index=False).assign_coords({'time':ds_lsts.time})
    ds_match = kaooi.match_filter(ds_whiten, dim='time', sampling_rate=sampling_rate, length='27.28s', carrier_freq=carrier_freq)
    ds_c = xrs.hilbert(ds_match, dim='time')
    ds_processed = ds_c.set_index({'time': ('longtime', 'shorttime')}).unstack()

    return ds_processed


def freq_whiten(x: Union[xr.DataArray, xr.Dataset], b: list, a: list, dim: str) -> Union[xr.DataArray, xr.Dataset]:
    '''
    take input xarray dataset or dataarray and frequency whiten along specified dimension

    Parameters
    ----------
    x : Union[xr.DataArray, xr.Dataset]
        input data to whiten
    b : np.array
        numerator coefficients of filter, used for magnitude of whitening
    a : np.array
        denominator coefficients of filter, used for magnitude of whitening
    dim : str
        dimension to whiten along
    '''
    # check if x is xr.DataArray or xr.Dataset
    if isinstance(x, xr.DataArray):
        return __freq_whiten_da(x,b,a,dim)
    elif isinstance(x, xr.Dataset):
        return x.map(__freq_whiten_da, args=(b, a, dim))
    else:
        raise ValueError(f'x must be an xarray.DataArray or xarray.Dataset, not {type(x)}')
        return

def freq_whiten_27(x: Union[xr.DataArray, xr.Dataset], dim: str) -> Union[xr.DataArray, xr.Dataset]:
    '''
    freq_whiten_27 - apply frequency whitening on 27.28 s segments individually

    Parameters
    ----------
    x : Union[xr.DataArray, xr.Dataset]
        input data to whiten
    dim : str
        dimension to whiten along
    '''
    b,a = signal.butter(4, [27 / (500 / 2), 33 / (500 / 2)], btype='band')
    return freq_whiten(x, b=b, a=a, dim=dim)


def __freq_whiten_da(
        x: xr.DataArray,
        b: list,
        a: list,
        dim: str
):
    '''
    __freq_whiten_da - maps of ::freq_whiten_da_chunk to all dask blocks

    Parameters
    ----------
    x : xr.DataArray
        input data to whiten
    b : np.array
        numerator coefficients of filter, used for magnitude of whitening
    a : np.array
        denominator coefficients of filter, used for magnitude of whitening
    dim : str
        dimension to whiten along
    
    Returns
    -------
    xw = xr.DataArray
        whitened xarray data
    '''
    xw = xr.map_blocks(__freq_whiten_da_chunk, x, args=(b, a, dim), template=x)
    return xw

def __freq_whiten_da_chunk(
        x: xr.DataArray,
        b: list,
        a: list,
        dim: str
):
    '''
    __freq_whiten_xr - mapping of ::__freq_whiten_np to take xarray.DataArray IO

    Parameters
    ----------
    x : xr.DataArray
        input data to whiten
    b : np.array
        numerator coefficients of filter, used for magnitude of whitening
    a : np.array
        denominator coefficients of filter, used for magnitude of whitening
    axis : int
        axis to whiten along
    '''
    axis = list(x.sizes.keys()).index(dim)
    return xr.DataArray(
        __freq_whiten_np(x.values, b, a, axis=axis),
        coords=x.coords,
        dims=x.dims
    )

def __freq_whiten_np(
        x: np.array,
        b: list,
        a: list,
        axis=0
    ) -> np.array:
    '''
    given numpy array of arbitray dimension, frequency whiten along specified axis

    Parameters
    ----------
    x : np.array
        input data to whiten
    b : np.array
        numerator coefficients of filter, used for magnitude of whitening
    a : np.array
        denominator coefficients of filter, used for magnitude of whitening
    axis : int
        axis to whiten along
    '''
    xf = scipy.fft.fft(x, axis=axis)
    x_phase = np.angle(xf)

    # compute unit pulse
    pulse = signal.unit_impulse(x.shape[axis], idx='mid')
    # Expand pulse to have length 1 in every dimension of x, except for the specified axis
    for dim in range(len(x.shape)):
        if dim != axis:
            pulse = np.expand_dims(pulse, dim)
    H = np.abs(scipy.fft.fft(signal.filtfilt(b, a, pulse, axis=axis), axis=axis))

    x_whiten_f = np.abs(H) * np.exp(1j * x_phase)
    x_whiten = scipy.fft.ifft(x_whiten_f, axis=axis).real

    return x_whiten
