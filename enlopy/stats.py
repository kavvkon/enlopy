"""This modules contains methods which correspond to estimation of statistics (features) for timeseries."""

import numpy as np
from scipy.signal import find_peaks
from itertools import groupby

#make it work only with ndarray?

def get_mean(x, trunced=False):
    if trunced:
        x = x[x>0]
    return np.mean(x)


def get_lf(x, trunced=False):
    """Load factor"""
    if trunced:
        x = x[x > 0]
    try:
        return np.mean(x) / np.max(x)
    except ZeroDivisionError:
        return np.nan

def get_trend(x, deg=1):
    # Assumes equally spaced series
    xx = np.arange(len(x))
    coeff = np.polyfit(xx, x, deg)
    return coeff[-2]

def get_rle(x, a):
    """Run length encoding algorithm.
    >>> get_rle("aaaaaaaaaaaaaaaaahhhh,,,,iuuuuuuiiaaalllaaa","a")
    [17, 3, 3]
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group))
               for value, group in groupby(x)
               if value == a]
        return res if len(res) > 0 else [0]

def largest_dur_of_zero(x):
    return max(get_rle(x, 0))

def get_peaks(x, min_distance=1, **kwargs):
    """
    Find peaks (local maxima) in a 1D array using scipy.signal.find_peaks.

    This implementation replaces the previous CWT-based approach for better
    efficiency and use of modern SciPy functions.

    Parameters:
        x (np.ndarray or list-like):
            The 1D input signal where peaks are to be found.
        min_distance (int):
            Minimum horizontal distance (in samples) required between
            neighboring peaks. Peaks closer than this distance are removed,
            keeping the highest one. Defaults to 1 (only compare immediate neighbors).
            This parameter loosely replaces the concept of scale/width (`n`)
            from the previous implementation.
        **kwargs:
            Additional keyword arguments passed directly to scipy.signal.find_peaks.
            Useful arguments include:
            - `height` (float or array-like): Minimum peak height.
            - `threshold` (float or array-like): Minimum vertical distance between peak and neighbors.
            - `prominence` (float or array-like): Minimum vertical distance peak stands out from surroundings.
            - `width` (float or array-like): Minimum peak width in samples.
            See the `scipy.signal.find_peaks` documentation for more details.

    Returns:
        np.ndarray: Indices of the peaks found in `x` that satisfy the conditions.

    Raises:
        ValueError: If the input array `x` is not 1D.
    """
    # Ensure input is a numpy array for compatibility and checks
    x_arr = np.asarray(x)

    if x_arr.ndim != 1:
        raise ValueError(f"Input array must be 1D, but got shape {x_arr.shape}")

    if len(x_arr) == 0:
        return np.array([], dtype=int) # Handle empty input gracefully

    # Use find_peaks. Pass min_distance as the distance parameter.
    # Allow users to override/add other parameters via kwargs.
    peaks_indices, _ = find_peaks(x_arr, distance=min_distance, **kwargs)

    return peaks_indices


def get_dur_val(x, a):
    """Get duration of periods that the timeseries is above one value.
    E.g It can be used to check the non-zero load duration"""
    return len(x[x==a]) # or sum(x[x>0]) ?

def get_percentile(x, a, trunced=False):
    if trunced:
        x = x[x>0]
    return np.percentile(x, a)


def get_ramp_rates(x, a=95):
    """Retrieve ramp rate of a percentile.
    a=100 for maximum-minimum. """
    ramps = np.diff(x)
    return np.percentile(ramps, 100 - a), np.percentile(ramps, a)


def get_highest_periodicity(x): #highest peaks of fft
    from scipy.signal import welch
    length = len(x)
    w = welch(x, scaling='spectrum', nperseg=length // 2)
    peak_ind = get_peaks(w[1], 5)
    periods = [np.round(1 / w[0][peak]) for peak in peak_ind]
    #filter periods. Remove too small and too high
    max_p = length // 2
    min_p = 3  # Do not show short term AR(1) - AR(3)
    return tuple(period for period in periods if min_p < period < max_p)

def get_load_ratio(x):
    if np.min(x) != 0:
        return np.max(x) / np.min(x)

def get_autocorr(x, lag=1):
    # We redefine AR with numpy to avoid importing statsmodels
    if np.isclose(np.min(x),np.max(x)):
        #if x vector is flat, autocorrelation is not defined
        return np.nan
    else:
        return np.corrcoef(np.array([x[0:len(x)-lag], x[lag:len(x)]]))[0,1]

# TODO: 2d: parcorr

from functools import partial
## Templates of stat collection can be found below:
all_stats_desc = {'Sum': np.sum,
                  'Average': get_mean,
                  'Max': np.max,
                  'Load Factor (peakiness)': get_lf,
                  'Total Zero load duration': partial(get_dur_val, a=0),
                  'Biggest duration of consecutive zero load': lambda x: np.max(get_rle(x, a=0)),
                  'Ramps (98%)': partial(get_ramp_rates, a=98),
                  'Min (2%)': partial(get_percentile, a=0.02, trunced=False),
                  'Periodicity': lambda x: get_highest_periodicity(x)[0:2],
                  'Autocorrelation(1)': partial(get_autocorr, lag=1),
                  'Trend': get_trend,
                  'Load ratio (max/min)': get_load_ratio,
                  'Num Prominent Peaks': partial(lambda x: len(get_peaks(x, prominence=np.std(x)/2))),
                  }
#to add more...
