import numpy as np
import pandas as pd

__all__ = ['make_timeseries', 'clean_convert']


def make_timeseries(x=None, year=2015, length=None, startdate=None, freq=None):
    """Convert numpy array to a pandas series with a timed index. Convenience wrapper around a datetime-indexed pd.DataFrame.

    Parameters:
        x: (nd.array) raw data to wrap into a pd.Series
        startdate: pd.datetime
        year: year of timeseries
        freq: offset keyword (e.g. 15min, H)
        length: length of timeseries
    Returns:
        pd.Series or pd.Dataframe with datetimeindex
    """

    if startdate is None:
        startdate = pd.datetime(year, 1, 1, 0, 0, 0)

    if x is None:
        if length is None:
            raise ValueError('The length or the timeseries has to be provided')
    else:  # if x is given
        length = len(x)
        if freq is None:
            # Shortcuts: Commonly used frequencies are automatically assigned
            if len(x) == 8760:
                freq = 'H'
            elif len(x) == 35040:
                freq = '15min'
            elif len(x) == 12:
                freq = 'm'
            else:
                raise ValueError('Input vector length must be 12, 8760 or 35040. Otherwise freq has to be defined')

    #enddate = startdate + pd.datetools.timedelta(seconds=_freq_to_sec(freq) * (length - 1) )
    date_list = pd.date_range(start=startdate, periods=length, freq=freq)
    if x is None:
        return pd.Series(np.nan, index=date_list)
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        x.index = date_list
        return x
    elif isinstance(x, (np.ndarray, list)):
        if len(x.shape) > 1:
            return pd.DataFrame(x, index=date_list)
        else:
            return pd.Series(x, index=date_list)
    else:
        raise ValueError('Unkown type of data passed')


def _freq_to_sec(freq_keyword):
    """ Converts pandas frequency string keyword to seconds. Not all frequency offsets can be converted to seconds.

    Arguments:
        freq_keyword: frequency based on pandas offsets
    Returns:
        int: corresponding seconds """
    from pandas.tseries.frequencies import to_offset

    try:
        return to_offset(freq_keyword).nanos * 1E-9
    except ValueError as e:
        raise ValueError('Works only with fixed frequencies e.g. h,s,t', e)


def human_readable_time(delta, terms=1):
    """Convert hours to human readable string

    Arguments:
        delta: time in seconds
        terms: how many word terms to use, to describe the timestep
    Returns
        str: Human readable string
    """
    # Inspired by http://stackoverflow.com/questions/26164671/convert-seconds-to-readable-format-time
    from dateutil.relativedelta import relativedelta

    intervals = ['years', 'months', 'days', 'hours', 'minutes', 'seconds']
    if delta > 31:
        delta = delta
    rd = relativedelta(hours=delta)
    out = ""
    for k in intervals[:terms]:
        if getattr(rd, k):
            out += '{} {} '.format(getattr(rd, k), k)
    return out.strip()


def clean_convert(x, force_timed_index=True, always_df=False, **kwargs):
    """Converts a list, a numpy array, or a dataframe to pandas series or dataframe, depending on the
    compatibility and the requirements. Designed for maximum compatibility.
    
    Arguments:
        x (list, np.ndarray): Vector or matrix of numbers. it can be pd.DataFrame, pd.Series, np.ndarray or list
        force_timed_index (bool): if True it will return a timeseries index
        year (int): Year that will be used for the index
        always_df (bool): always return a dataframe even if the data is one dimensional
        **kwargs: Exposes arguments of :meth:`make_timeseries`
    Returns:
        pd.Series: Timeseries
        
    """

    if isinstance(x, list):  # nice recursions
        return clean_convert(pd.Series(x), force_timed_index, always_df, **kwargs)

    elif isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            return clean_convert(pd.Series(x),force_timed_index, always_df, **kwargs)
        else:
            return clean_convert(pd.DataFrame(x), force_timed_index, always_df, **kwargs)

    elif isinstance(x, pd.Series):
        if always_df:
            x = pd.DataFrame(x)
        if x.index.is_all_dates:
            return x
        else:  # if not datetime index
            if force_timed_index:
                print('Forcing Datetimeindex into passed timeseries.'
                      'For more accurate results please pass a pandas time-indexed timeseries.')
                return make_timeseries(x, **kwargs)
            else:  # does not require datetimeindex
                return x

    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1 and not always_df:
            return clean_convert(x.squeeze(), force_timed_index, always_df, **kwargs)
        else:
            if force_timed_index and not x.index.is_all_dates:
                return make_timeseries(x, **kwargs)
            else:  # does not require datetimeindex
                return x
    else:
        raise ValueError('Unrecognized Type. Has to be one of the following: pd.DataFrame, pd.Series, np.ndarray or list')

