'''
Usefull Tools for Kauai beacon analysis with OOI Hydrophones
'''
import pandas as pd
from typing import Optional
import numpy as np
from scipy import signal
import xarray as xr
import kaooi


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


def __keytimes_2023(future: bool) -> list:
    '''
    returns only key times for the year 2023
    manually coded to reflect the schedule of the Kauai source
    Parameters
    ----------
    future : bool
        If True, return the future keytimes, else return up to current day
    '''

    # Transmission details at marasondo.org
    tx_days = {
        '01': [],
        '02': [],
        '03': [18, 22, 26, 30],
        '04': [3,7,11,15,19,23,27],
        '05': [1,5,9,13,17,21,25,29],
        '06': [2,6,10,14,18,22,26],
        '07': [],
        '08': [3,7,11,15,19,23,27,31],
        '09': [4,8,12,16,20,24,28],
        '10': [2,6,10,14,22,26,30],
        '11': [3,7,11,15,19,23,27],
        '12': []
    }


    Tx_times = []

    for month in tx_days:
        for day in tx_days[month]:
            time = pd.Timestamp(f'2023-{month}-{day:02}')
            
            
            # full day skips
            if time == pd.Timestamp('2023-10-18 00:00:00'):
                continue
            # don't include future transmissions
            if (not future) & (time > pd.Timestamp('today')):
                continue
            
            Tx_times.append(time)
            Tx_times.append(time + pd.Timedelta('4H'))
            Tx_times.append(time + pd.Timedelta('8H'))
            Tx_times.append(time + pd.Timedelta('12H'))
            Tx_times.append(time + pd.Timedelta('16H'))
            Tx_times.append(time + pd.Timedelta('20H'))
            
    # remove failed transmissions
    Tx_times.remove(pd.Timestamp('2023-04-19 04:00:00'))
    Tx_times.remove(pd.Timestamp('2023-06-26 04:00:00'))
    Tx_times.remove(pd.Timestamp('2023-06-26 08:00:00'))
    Tx_times.remove(pd.Timestamp('2023-06-26 12:00:00'))
    Tx_times.remove(pd.Timestamp('2023-06-26 16:00:00'))
    Tx_times.remove(pd.Timestamp('2023-06-26 20:00:00'))
    Tx_times.remove(pd.Timestamp('2023-09-24 04:00:00'))
    return Tx_times


def construct_mseq() -> np.ndarray:
    '''
    construct_mseq - construct the m-sequence encoding

    Returns
    -------
    seq : np.ndarray
        m-sequence encoding for octal law 3471
    '''

    # 10 bit shift register
    # initial state of 0000000001
    #  3 / 4 / 7 / 1
    # 011 100 111 001
    # [0,3,4,5,8,9,10] (indices of taps)
    seq, _ = signal.max_len_seq(10, state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], taps=[0, 3, 4, 5, 8, 9, 10])
    return seq


def construct_replica_aliased(
    sampling_rate: float,
    carrier_freq: float = 75,
    q: int = 2,
    verbose: bool = False,
) -> xr.DataArray:
    '''
    construct_replica - construct replica of signal broadbcast by Kauai source
        for given sampling frequency. there is no consideration to aliasing. To
        generate a signal that is not aliased, use construct_replica

    Parameters
    ----------
    sampling_rate : float
        sampling rate of replica, in Hz
    carrier_freq : float
        carrier frequency of replica, in Hz
    q : int
        quality factor (number of cycles per bit)
    verbose : bool
        if True, print info

    Returns
    -------
    replica : xr.DataArray
        replica of signal
    '''

    m = construct_mseq()

    # width of single bit in seconds
    bit_width = q / carrier_freq
    if verbose:
        print('bit_width', bit_width)

    # length of sequence in seconds
    seq_len = len(m) * bit_width
    if verbose:
        print('seq_len', seq_len)

    # number of samples in sequence
    seq_len_samples = seq_len * sampling_rate
    if seq_len_samples % 1 != 0:
        raise Exception(
            f'sequency length of {seq_len}s cannot be represented by integer number of samples for sampling rate of {sampling_rate} Hz. This is required for circular stacking'
        )
    else:
        seq_len_samples = int(seq_len_samples)
    if verbose:
        print('seq_len_samples', seq_len_samples)

    # number of samples per bit (not necissarily an integer)
    samples_per_bit = (bit_width) / (1 / sampling_rate)
    if verbose:
        print('samples_per_bit', samples_per_bit)

    ## construct binary sequence
    bin_seq = np.ones(seq_len_samples)
    a = 0

    for k in range(len(m)):
        b = int(np.floor((k + 1) * samples_per_bit))
        bin_seq[a : b + 1] = bin_seq[a : b + 1] * m[k]
        a = b

    # change from 1s and 0s to -1 and 1
    bin_seq = -2 * bin_seq + 1

    t = np.arange(0, seq_len, 1 / sampling_rate)[: len(bin_seq)]

    carrier = np.sin(2 * np.pi * carrier_freq * t)
    replica = xr.DataArray(carrier * bin_seq, dims=['time'], coords={'time': t}, name='signal replica')

    return replica


def construct_replica(
    sampling_rate: float,
    carrier_freq: float = 75,
    q: int = 2,
    verbose: bool = False,
) -> xr.DataArray:
    '''
    construct_replica - construct replica of signal broadbcast by Kauai source
        for given sampling frequency.
    to handle low sampling rates relative to signal bandwidth, the following is done
    - for sampling rates greater than 1000 Hz, the replica is constructed as normal
        - sides lobes above 1000Hz are well below -100dB from signal max
    - for sampling rates lower than 1000 Hz, replica is constructed at 1000 Hz and then
        resampled to desired sampling rate. scipy.signal.decimate is used and the anti-aliasing
        filter is set to be zero_phase.

    Parameters
    ----------
    sampling_rate : float
        sampling rate of replica, in Hz
    carrier_freq : float
        carrier frequency of replica, in Hz
    q : int
        quality factor (number of cycles per bit)
    verbose : bool
        if True, print info

    Returns
    -------
    replica : xr.DataArray
        replica of signal
    '''
    if sampling_rate < 1000:
        replica_1000_np = construct_replica_aliased(1000, carrier_freq, q, verbose).values
        replica_np = __decimate_nonint(replica_1000_np, 1000, sampling_rate)

        # get time coordinate
        m = construct_mseq()

        # width of single bit in seconds
        bit_width = q / carrier_freq
        if verbose:
            print('bit_width', bit_width)

        # length of sequence in seconds
        seq_len = len(m) * bit_width

        t = np.arange(0, seq_len, 1 / sampling_rate)[: len(replica_np)]

        replica = xr.DataArray(replica_np, dims=['time'], coords={'time': t}, name='signal replica')
    else:
        replica = construct_replica_aliased(sampling_rate, carrier_freq, q, verbose)

    return replica


def __decimate_nonint(input_array: np.array, old_sampling_rate: float, new_sampling_rate: float) -> np.array:
    '''
    decimate signal by noninteger factor

    Parameters
    ----------
    input_array : np.ndarray
        input signal
    old_sampling_rate : float
        sampling rate of input signal
    new_sampling_rate : float
        desired sampling rate of output signal

    Returns
    -------
    output_array : np.ndarray
        decimated signal
    '''
    if old_sampling_rate % new_sampling_rate == 0:
        # If the sampling rates have an integer ratio, use regular decimate
        decimation_factor = old_sampling_rate // new_sampling_rate
        return signal.decimate(input_array, decimation_factor, zero_phase=True)
    else:
        # Find the least common multiple of the old and new sampling rates
        lcm = np.lcm(old_sampling_rate, new_sampling_rate)

        # Upsample to the least common multiple sampling rate
        upsampled_array = signal.resample(input_array, int(lcm / old_sampling_rate * len(input_array)))

        # Decimate to the new sampling rate
        decimation_factor = int(lcm / new_sampling_rate)
        output_array = signal.decimate(upsampled_array, decimation_factor, zero_phase=True)
        return output_array


def add_ltst_coords(ds: xr.Dataset, dim: str, sampling_rate: float, length: str = '2H'):
    '''
    add longtime shorttime coords. Adds longtime/shorttime coordinates
        relative to the 27.28 second signal replica length and the supplied
        sampling rate.

    Parameters
    ----------
    ds : xr.Dataset
        dataset to add lt / st coords to
    dim : str
        dimension to add coords to
    sampling_rate : float
        sampling rate of data in dimension 'dim'
    length : str
        length of dataset in dimension dim, should be readable by pd.Timedelta
    '''

    replica = kaooi.construct_replica(sampling_rate=sampling_rate)

    # construct new time coordinates
    len_rep = len(replica) / sampling_rate
    lt = np.arange(0, pd.Timedelta(length).value / 1e9, len_rep)[:-1]
    st = np.arange(0, len_rep, 1 / sampling_rate)
    ltlt, stst = np.meshgrid(lt, st)

    stst = stst.T.flatten()
    ltlt = ltlt.T.flatten()

    # keep only integer number of replicas in ds
    ds = ds.isel({dim: slice(0, int(ds[dim].size / replica[dim].size) * replica[dim].size)})

    ds = ds.assign_coords({'longtime': (dim, ltlt), 'shorttime': (dim, stst)})
    
    # Can't be supported with netcdf ...
    #ds = ds.set_index(
    #    {'time': ('longtime', 'shorttime')}
    #)
    return ds
