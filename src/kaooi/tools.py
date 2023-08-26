'''
Usefull Tools for Kauai beacon analysis with OOI Hydrophones
'''
import pandas as pd
from typing import Optional
import xarray as xr
from tqdm import tqdm
import ooipy
import numpy as np
import obspy

# hydrophone node names
#LJ01A has been removed because of fragmented data (could be added back if OOI changes raw data server)
bb_nodes = ['LJ01D', 'PC01A', 'PC03A', 'LJ01C', 'LJ03A']
lf_nodes = ['AXBA1', 'AXCC1', 'AXEC2', 'HYS14', 'HYSB1']


def get_Tx_keytimes(year: int, future: Optional[bool] = False) -> list:
    '''
    get_Tx_keytimes - given the year, return a list of pd.Timestamps that indicate each time
    the Kauai source transmitted for a given year

    Parameters
    ----------
    year : int
        year to get key times for. Currently supported years: [2023]
    future : bool
        If True, return the future keytimes, else return up to current day
    '''
    if year == 2023:
        return __keytimes_2023(future)

def __keytimes_2023( future : bool) -> list:
    '''
    returns only key times for the year 2023
    manually coded to reflect the schedule of the Kauai source
    '''
    
    ## period before amplifier failure
    section1_days = list(pd.date_range(pd.Timestamp('2023-03-18'), pd.Timestamp('2023-06-25'), freq='4D'))

    # remove day level failures
    section1_days.remove(pd.Timestamp('2023-03-22'))
    
    # construct list of times (6 per day)
    section1 = []
    for time in section1_days:
        section1.append(time)
        section1.append(time + pd.Timedelta('4H'))
        section1.append(time + pd.Timedelta('8H'))
        section1.append(time + pd.Timedelta('12H'))
        section1.append(time + pd.Timedelta('16H'))
        section1.append(time + pd.Timedelta('20H'))

    # remove Txs that didn't happen:
    section1.remove(pd.Timestamp('2023-04-19 04:00:00'))

    ## period after amplifier failure
    if future:
        section2_days = list(pd.date_range(pd.Timestamp(
            '2023-08-03'), pd.Timestamp('2024-01-01'), freq='4D'))
    else:
        section2_days = list(pd.date_range(pd.Timestamp('2023-08-03'), pd.Timestamp('today'), freq='4D'))

    # construct list of times (6 per day)
    section2 = []
    for time in section2_days:
        section2.append(time)
        section2.append(time + pd.Timedelta('4H'))
        section2.append(time + pd.Timedelta('8H'))
        section2.append(time + pd.Timedelta('12H'))
        section2.append(time + pd.Timedelta('16H'))
        section2.append(time + pd.Timedelta('20H'))

    # future updates to schedule can be added below:

    return section1 + section2

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

    for node in tqdm(lf_nodes, disable=~verbose):
        if verbose:
            print(f'{node}:')
        hdata[node] = ooipy.get_acoustic_data_LF(
            start_time,
            end_time,
            node,
            verbose=verbose
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
        chunk_sizes: Optional[dict] = {'time': 125e3, 'transmission': 1},
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
    chunk_sizes : dict
        dictionary of chunk sizes for each dimension of the dataset

    Returns
    -------
    ds : xr.Dataset
        dataset of hydrophone data
    '''

    hdata_x = {}
    for node in hdata:
        if hdata[node] == None:
            nsamples = int(pd.Timedelta(length).value/1e9*1000 + 1)
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
            hdata_x[node].attrs['processing'] = str(hdata_x[node].attrs['processing'])
    try:
        return xr.Dataset(hdata_x).chunk(chunk_sizes).drop_indexes(coord_names=['transmission'])
    except ValueError:
        print(f'uneven data coverage for {key_time}')
        return None