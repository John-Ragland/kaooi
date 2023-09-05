'''
Usefull Tools for Kauai beacon analysis with OOI Hydrophones
'''
import pandas as pd
from typing import Optional
import numpy as np
from scipy import signal
import xarray as xr

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
    seq,_ = signal.max_len_seq(10, state=[0,0,0,0,0,0,0,0,0,1], taps=[0,3,4,5,8,9,10])
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
    bit_width = q/carrier_freq
    if verbose: print('bit_width', bit_width)
    
    # length of sequence in seconds
    seq_len = len(m)*bit_width
    if verbose: print('seq_len', seq_len)
    
    # number of samples in sequence
    seq_len_samples = seq_len*sampling_rate
    if seq_len_samples % 1 != 0:
        raise Exception(f'sequency length of {seq_len}s cannot be represented by integer number of samples for sampling rate of {sampling_rate} Hz. This is required for circular stacking')
    else:
        seq_len_samples = int(seq_len_samples)
    if verbose: print('seq_len_samples', seq_len_samples)
    
    # number of samples per bit (not necissarily an integer)
    samples_per_bit = (bit_width)/(1/sampling_rate)
    if verbose: print('samples_per_bit', samples_per_bit)

    ## construct binary sequence
    bin_seq = np.ones(seq_len_samples)
    a = 0

    for k in range(len(m)):
        b = int(np.floor((k+1)*samples_per_bit))
        bin_seq[a:b+1] = bin_seq[a:b+1]*m[k]
        a=b

    # change from 1s and 0s to -1 and 1
    bin_seq = -2*bin_seq + 1

    t = np.arange(0, seq_len, 1/sampling_rate)[:len(bin_seq)]

    carrier = np.sin(2*np.pi*carrier_freq*t)
    replica = xr.DataArray(carrier*bin_seq, dims=['time'], coords={'time':t}, name='signal replica')

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
        replica_1000_np= construct_replica_aliased(1000, carrier_freq, q, verbose).values
        replica_np = __decimate_nonint(replica_1000_np, 1000, sampling_rate)
        
        # get time coordinate
        m = construct_mseq()

        # width of single bit in seconds
        bit_width = q/carrier_freq
        if verbose: print('bit_width', bit_width)
    
        # length of sequence in seconds
        seq_len = len(m)*bit_width

        t = np.arange(0, seq_len, 1/sampling_rate)[:len(replica_np)]

        replica = xr.DataArray(replica_np, dims=['time'], coords={'time':t}, name='signal replica')
    else:
        replica = construct_replica_aliased(sampling_rate, carrier_freq, q, verbose)
    
    return replica


def __decimate_nonint(input_array, old_sampling_rate, new_sampling_rate):
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