'''
detection.py functions towards detecting the kauai signal in data
'''
import kaooi
import xarray as xr
import numpy as np
from scipy import signal
from tqdm import tqdm

def estimate_peaks_array(da, noise_slice=(0, None), snr_threshold =10):
    '''
    estimate_peaks - given processed data, estimate location and snr of peaks

    Parameters
    ----------
    da : xr.DataArray
        dataset of processed, stacked, complex envelope of data. Should have dimensions ['shorttime', 'transmission']
    noise_slice : slice
        slice in short time that contains just noise (default is (0, None), i.e. entire slice)
    snr_threshold : float
        threshold for SNR in dB. Default is 10
    '''

    noise_rms = np.sqrt(((da.sel({'shorttime': slice(0,None)}))**2).mean('shorttime'))
    snr = 20 * np.log10(( da / noise_rms))

    peak_locs = []
    peak_heights = []

    for k in range(snr.sizes['transmission']):
        loc, height = signal.find_peaks(snr.isel({'transmission': k}), height=snr_threshold)
        peak_locs.append(list(loc))
        peak_heights.append(list(height['peak_heights']))

    peak_locs_s = []
    for k in range(len(peak_locs)):
        peak_locs_s.append(da.shorttime.values[peak_locs[k]].tolist())

    arrival_times =  {
        'peak_locs': peak_locs_s,
        'peak_heights': peak_heights,
        'transmission': da.transmission.values.tolist(),
    }
    return arrival_times


def get_peak_heights(da: xr.DataArray, noise_slice: slice, circular_avg: int, snr_threshold: float = 9) -> dict:
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
    snr_threshold : float
        threshold for SNR to consider a peak

    Returns
    -------
    peaks : dict
        dictionary with keys ['locations, 'heights'].
        containing a list of numpy arrays for each long-time coordinate

    '''

    rms = np.sqrt(
        (np.abs(da.rolling(longtime=circular_avg).mean() ** 2)).sel({'shorttime': noise_slice}).mean('shorttime')
    )
    snr_stacked = 20 * np.log10((np.abs(da.rolling(longtime=circular_avg).mean() / rms)))

    locs = []
    heights = []
    longtimes = []

    for k in range(snr_stacked.sizes['longtime']):
        loc, height = signal.find_peaks(snr_stacked[k, :].values.flatten(), height=snr_threshold)

        locs.append(loc)
        heights.append(height['peak_heights'])
        longtimes.append(np.ones(len(loc)) * da.longtime.values[k])

    locs = np.concatenate(locs)
    heights = np.concatenate(heights)
    longtimes = np.concatenate(longtimes)

    peaks = {'shorttime': da.shorttime.values[locs], 'longtime': longtimes, 'snr': heights}
    return peaks


def estimate_arrival_time(
    peak_heights: dict,
):
    '''
    estimate_arrival_time - given peak heights computed with ::get_peak_heights::,
        estimate the snr weighted, average arrival time

    Parameters
    ----------
    peak_heights : dict
        dictionary with keys ['shorttime, 'longtime', 'snr'], and associated numpy arrays.
        peak_heights is created with ::get_peak_heights::

    Returns
    -------
    arrival_time : float
        estimated, snr weighted arrival time
    '''

    snr_lin = 10 ** (peak_heights['snr'] / 20)
    arrival_time = np.sum(peak_heights['shorttime'] * snr_lin) / np.sum(snr_lin)
    return arrival_time


def create_at_scatter(peak_heights: list, transmissions: list):
    '''
    create at scatter - give list of peak_heights, create variables transmission, shorttime, and snr
        that can be plotted in a scatter plot. There are multiple peaks per transmission

    Parameters
    ----------
    peak_heights : list
        list of dictionaries with keys ['shorttime, 'longtime', 'snr'], and associated numpy arrays.
        single peak_height is created with ::get_peak_heights::, and represents a single transmission
    transmissions : list
        list of transmission times, in for ::pd.Timestamp::

    Returns
    -------
    transmission_times : list
        list of transmission times, repeated for each peak
    arrival_times : list
        list of arrival times, for each peak
    snrs : list
        list of snrs, for each peak
    '''

    transmission_times = []
    arrival_times = []
    snrs = []
    for k in range(len(peak_heights)):
        transmission_times += [transmissions[k]] * len(peak_heights[k]['shorttime'])
        arrival_times += list(peak_heights[k]['shorttime'])
        snrs += list(peak_heights[k]['snr'])

    return transmission_times, arrival_times, snrs
