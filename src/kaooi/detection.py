'''
detection.py functions towards detecting the kauai signal in data
'''
import kaooi
import xarray as xr
import numpy as np
from scipy import signal

def get_peak_heights(
    da : xr.DataArray,
    noise_slice : slice,
    circular_avg : int,
    )-> dict:
    '''
    get_peak_heights - given a match filtered transmission (with dimensions longtime / shorttime),
        compute the arrivals throughout the transmission. Returns a list of arrivals for every
        long-time bin
    
    Parameters
    ----------
    da : xr.DataArray
        single transmission with dimensions longtime and shorttime
    noise_slice : slice
        slice in short time to consider only noise for SNR calculation
    circular_avg : int
        number of circular averages to compute
    
    Returns
    -------
    peaks : dict
        dictionary with keys ['locations, 'heights'].
        containing a list of numpy arrays for each long-time coordinate
        
    '''
    
    rms = np.sqrt((np.abs(da.rolling(longtime=circular_avg).mean()**2)).sel({'shorttime':noise_slice}).mean('shorttime'))
    snr_stacked = 20*np.log10((np.abs(da.rolling(longtime=circular_avg).mean()/rms)))
    
    locs = []
    heights = []
    longtimes = []

    for k in range(snr_stacked.sizes['longtime']): 
        loc, height = signal.find_peaks(snr_stacked[k,:].values.flatten(), height=7)

        locs.append(loc)
        heights.append(height['peak_heights'])
        longtimes.append(np.ones(len(loc))*da.longtime.values[k])

    locs = np.concatenate(locs)
    heights = np.concatenate(heights)
    longtimes = np.concatenate(longtimes)

    peaks = {
        'shorttime':da.shorttime.values[locs],
        'longtime': longtimes,
        'snr':heights
    }
    return peaks