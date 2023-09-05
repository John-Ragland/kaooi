'''
routines for downloading Kauai transmission data from OOI
'''
import pandas as pd
from typing import Optional
import xarray as xr
from tqdm import tqdm
import ooipy
import numpy as np
import obspy
import os

# import all functions from tools
from .tools import *
from .signal_processing import *

# hydrophone node names
#LJ01A has been removed because of fragmented data (could be added back if OOI changes raw data server)
bb_nodes = ['LJ01D', 'PC01A', 'PC03A', 'LJ01C', 'LJ03A']

lf_nodes = ['AXAS1', 'AXAS2', 'AXBA1', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXID1', 'HYS11', 'HYS12', 'HYS13', 'HYS14', 'HYSB1']
lfhyd_nodes = ['AXBA1','AXCC1','AXEC2','HYS14','HYSB1']

# construct list of lf devices (hydrophones and seismometers)
devices = []
for node in lf_nodes:
    if node in lfhyd_nodes:
        devices.append({'node':node, 'channel':'HDH'})
        devices.append({'node':node, 'channel':'HNZ'})
    else:
        devices.append({'node':node, 'channel':'EHZ'})

def downloadTx_200hz(
        Tx_time: pd.Timestamp,
        ds_dir: str = '/datadrive/kauai/transmissions/200Hz_sensors/',
        length: str = '2H',
        verbose=False,) -> None:
    '''
    download and save data from single transmission for all sensors. And save the data to disk.

    Parameters
    ----------
    Tx_time : pd.Timestamp
        start time of transimission
    ds_dir : str    
        directory to save data to. data org inside of directory should have real and imag folders
    length : str
        length of data to download, should be readable by pd.Timedelta
    verbose : bool
        if True, print progress
    '''

    # check if data already exists
    fnr = f'{ds_dir}/real/{Tx_time.strftime("%Y%m%dT%H%M%S")}.nc'
    fni = f'{ds_dir}/imag/{Tx_time.strftime("%Y%m%dT%H%M%S")}.nc'

    if os.path.exists(fnr) & os.path.exists(fni):
        if verbose: print(f'{Tx_time.strftime("%Y%m%dT%H%M%S")} skipped')
        return

    #download data from ooi server
    data = kaooi.download_data_lf(Tx_time, length=length, verbose=verbose)

    # convert to x array.dataset
    data_x = kaooi.construct_xds(Tx_time, data, length=length, sampling_rate=200, chunk_sizes={'time':1440001})
    
    # skip if there is uneven data coverage
    if data_x is None:
        if verbose: print(f'{Tx_time.strftime("%Y%m%dT%H%M%S")} skipped')
        return
    
    # match filter the data
    data_x_match = kaooi.match_filter(data_x, dim='time', sampling_rate=200, length=length)

    # save to disk
    # duct taping that netcdf doesn't allow complex numbers    
    data_x_match.real.to_netcdf(fnr)
    data_x_match.imag.to_netcdf(fni)

    if verbose: print(f'{Tx_time.strftime("%Y%m%dT%H%M%S")} completed.')
    return

def download_data_bb(
        key_time : pd.Timestamp,
        length: Optional[str] = '2H',
        verbose: Optional[bool] = True) -> dict:
    '''
    download_data - given start time return xr.Dataset of hydrophone for every BB hydrophone
    two hours of data will be downloaded
    
    Parameters
    ----------
    key_time : pd.Timestamp
        start time of transimission
    length : str
        length of data to download, should be readable by pd.Timedelta
    verbose : bool
        if True, show progress bar
    
    Returns
    -------
    ds : xr.Dataset
        dataset of hydrophone data
    '''
    start_time = key_time.to_pydatetime()
    end_time = start_time + pd.Timedelta(length)


    hdata = {}

    for node in tqdm(bb_nodes, disable=~verbose):
        if verbose:
            print(f'{node}:')
        hdata[node] = ooipy.get_acoustic_data(start_time, end_time, node, verbose=verbose)

    return hdata

def download_data_lf(
        key_time : pd.Timestamp,
        length: Optional[str] = '2H',
        verbose: Optional[bool] = True) -> dict:
    '''
    download_data - given start time return xr.Dataset of hydrophone for every LF hydrophone
    two hours of data will be downloaded
    
    Parameters
    ----------
    key_time : pd.Timestamp
        start time of transimission
    length : str
        length of data to download, should be readable by pd.Timedelta
    verbose : bool
        if True, show progress bar
    
    Returns
    -------
    ds : xr.Dataset
        dataset of hydrophone data
    '''
    start_time = key_time.to_pydatetime()
    end_time = start_time + pd.Timedelta(length)


    hdata = {}

    for device in tqdm(devices, disable=~verbose):
        if verbose:
            print(f"{device['node']}.{device['channel']}:")

        hdata[f"{device['node']}.{device['channel']}"] = ooipy.get_acoustic_data_LF(
            start_time,
            end_time,
            device['node'],
            verbose=verbose,
            channel=device['channel'],
            zero_mean=True
        )

    return hdata

def get_ooi_data(
        key_time : pd.Timestamp,
        length: Optional[str] = '2H',
        verbose: Optional[bool] = True,
        chunk_sizes: Optional[dict] = {'time':125e3, 'transmission':1}) -> xr.Dataset:
    '''
    get_ooi_data - given start time return xr.Dataset of hydrophone for every hydrophone
    two hours of data will be downloaded
    
    Parameters
    ----------
    key_time : pd.Timestamp
        start time of transimission
    length : str
        length of data to download, should be readable by pd.Timedelta
    verbose : bool
        if True, show progress bar
    
    Returns
    -------
    ds : xr.Dataset
        dataset of hydrophone data
    '''
    start_time = key_time.to_pydatetime()
    end_time = start_time + pd.Timedelta(length)

    # download broadband data
    hdata_bf = download_data_bb(key_time, length, verbose)
    # decimate broadband data to fs = 1000
    hdata_bf = decimate_data(hdata_bf)

    # download low frequency data
    hdata_lf = download_data_lf(key_time, length, verbose)
    # upsample low frequency data to fs = 1000
    hdata_lf = resample_data(
        hdata_lf,
        fs=1000
    )

    # slice low frequency data to exactly match broadband data
    for node in hdata_lf:
        try:
            hdata_lf[node] = hdata_lf[node].slice(
                starttime=obspy.core.UTCDateTime(start_time),
                endtime=obspy.core.UTCDateTime(end_time)
            )
        except AttributeError:
            pass

    hdata = hdata_bf | hdata_lf

    # check if there is partial data and remove it
    for node in hdata:
        if hdata[node] == None:
            pass
        else:
            if hdata[node].stats['endtime'] != end_time:
                hdata[node] = None
        
    hdata_x = construct_xds(key_time, hdata, length, chunk_sizes=chunk_sizes)
    return hdata_x

def decimate_data(hdata: dict):
    '''
    decimate_data decimate broadband data from fs=64khz to fs=1000Hz
    done in two ds factors of 8 to avoid unstable filters

    TODO there seems to be a problem with the beginning of data samples
    '''
    for node in hdata:
        if hdata[node] == None:
            pass
        else:
            hdata[node].decimate(8)
            hdata[node].decimate(8)
    return hdata

def resample_data(hdata: dict, fs: int = 1000) -> dict:
    '''
    resample_data resample broadband data to a new sampling frequency
    '''
    for node in hdata:
        if hdata[node] == None:
            pass
        else:
            hdata[node].resample(fs)
    return hdata

def construct_xds(
        key_time: pd.Timestamp,
        hdata: dict,
        length: str,
        sampling_rate: float,
        chunk_sizes: Optional[dict] = None,
    ) -> xr.Dataset:
    '''
    construct_xds - construct xarray dataset of hydrophone data for single transmission

    Parameters
    ----------
    key_time : pd.Timestamp
        start time of transimission
    hdata : dict
        dictionary of hydrophone data
    length : str
        length of data to download, should be readable by pd.Timedelta
    sampling_rate : float
        if included, then time coordinate is constructed and added to dataset
    chunk_sizes : dict
        dictionary of chunk sizes for each dimension of the dataset


    Returns
    -------
    ds : xr.Dataset
        dataset of hydrophone data
    '''
    if sampling_rate is not None:
        time_coord = np.arange(0,pd.Timedelta(length).value/1e9+1/sampling_rate,1/sampling_rate)

    hdata_x = {}
    for node in hdata:
        if hdata[node] == None:

            nsamples = int(pd.Timedelta(length).value/1e9*sampling_rate + 1)
            single_channel = np.expand_dims(np.ones(nsamples)*np.nan, 1)

            hdata_x[node] = xr.DataArray(
                single_channel,
                dims=['time', 'transmission'],
                coords={'transmission':[key_time]},
                name=node)
        else:
            single_channel = np.expand_dims(hdata[node].data, 1)

            hdata_x[node] = xr.DataArray(
                single_channel,
                dims=['time', 'transmission'],
                coords={'transmission':[key_time]},
                name=node,
                attrs=hdata[node].stats)
            
            # change attributes to strings
            hdata_x[node].attrs['starttime'] = hdata_x[node].attrs['starttime'].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            hdata_x[node].attrs['endtime'] = hdata_x[node].attrs['endtime'].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            hdata_x[node].attrs['mseed'] = str(hdata_x[node].attrs['mseed'])
            try:
                hdata_x[node].attrs['processing'] = str(hdata_x[node].attrs['processing'])
            except KeyError:
                pass
    try:
        ds = xr.Dataset(
            hdata_x,
            coords={'time':time_coord}
        )

        # chunk the dataset
        if chunk_sizes is None:
            chunk_sizes = {'transmission':1, 'time':ds.sizes['time']}

        ds = ds.chunk(chunk_sizes).drop_indexes(coord_names=['transmission'])
        return ds
    
    except ValueError as e:
        print(e)
        print(f'uneven data coverage for {key_time}')
        return None
