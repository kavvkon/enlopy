from __future__ import division
import numpy as np
import pandas as pd

from .utils import clean_convert

__all__ = ['reshape_timeseries', 'get_LDC', 'get_load_archetypes', 'get_load_stats', 'detect_outliers']

def reshape_timeseries(Load, x='dayofyear', y=None, aggfunc='sum'):
    """Returns a reshaped pandas DataFrame that shows the aggregated load for selected
    timeslices. e.g. time of day vs day of year

    Parameters:
        Load (pd.Series, np.ndarray): timeseries
        x (str): x axis aggregator. Has to be an accessor of pd.DatetimeIndex
         (year, dayoftime, week etc.)
        y (str): similar to above for y axis
    Returns:
        reshaped pandas dataframe according to x,y
    """

    # Have to convert to dataframe in order for pivottable to work
    # 1D, Dataframe
    a = clean_convert(Load.copy(), force_timed_index=True, always_df=True)
    a.name = 0
    if len(a.columns) > 1:
        raise ValueError('Works only with 1D')

    if x is not None:
        a[x] = getattr(a.index, x)
    if y is not None:
        a[y] = getattr(a.index, y)
    a = a.reset_index(drop=True)

    return a.pivot_table(index=x, columns=y,
                         values=a.columns[0],
                         aggfunc=aggfunc).T


def get_LDC(Load, x_norm=True, y_norm=False):
    """Generates the Load Duration Curve based on a given load. For 2-dimensional dataframes the x-axis sorting
     is done based on sum of all series. Sorting on the y-axis is done based on the coefficient of variance.

    Arguments:
        Load (pd.Series): timeseries
        x_norm (bool): Normalize x axis (0,1)
        y_norm (bool): Normalize y axis (0,1)
    Returns:
        np.ndarray: tuple (x, y) ready for plotting (e.g. plt(\*LDC_load(load)))
    """
    Load1 = clean_convert(Load)
    if Load1.ndim >= 2:
        # Sort x axis by total value
        sorted_ind = Load1.sum(axis=1).sort_values(ascending=False).index
        # Sort series by variance coefficient. Baseline load should have smaller CV and should go lower
        sorted_cols = (Load1.std()/Load1.mean()).sort_values().index
        y = Load1.loc[sorted_ind, sorted_cols].values
    else:
        y = Load1.sort_values(ascending=False).values
    x = np.arange(1, len(y) + 1 )
    if x_norm:
        x = x / len(x)
    if y_norm:
        y = y / y.max()
    return x, y

    # remove nan because histogram does not work
    #load_masked = Load[~np.isnan(Load)]
    #n, xbins = np.histogram(load_masked, bins=bins, density=True)
    # xbins = xbins[:-1] #remove last element to make equal size
    #cum_values = np.zeros(xbins.shape)
    #cum_values[1:] = np.cumsum(n*np.diff(xbins))
    #out = np.array([1-cum_values, xbins])
    # out = np._r[[1 0], out] # Add extra point
    #if trunc_0: # Trunc non zero elements
    #    out[out < 0] = 0
    #return out


def get_load_archetypes(Load, k=2, x='hour', y='dayofyear', plot_diagnostics=False):
    """Extract typical load profiles using k-means and vector quantization. the time scale of archetypes depend on the selected dimensions (x,y).
    For the default values daily archetypes will be extracted.

    Parameters:
        Load (pd.Series): timeseries
        k (int): number of archetypes to identify and extract
        x (str): This will define how the timeseries will be grouped by. Has to be an accessor of pd.DatetimeIndex
        y (str): similar to above for y axis.
        plot_diagnostics (bool): If true a figure is plotted showing an overview of the results
    Returns:
        np.ndarray: dimensions (k, len(x))
    """
    from scipy.cluster.vq import whiten, kmeans, vq

    df = reshape_timeseries(Load, x=x, y=y, aggfunc='mean')
    df_white = whiten(df.astype(float))
    clusters_center, __ = kmeans(df_white, k)

    if plot_diagnostics:
        try:
            import matplotlib.pyplot as plt
            clusters, _ = vq(df_white, clusters_center)
            cm = _n_colors_from_colormap(k)
            df.T.plot(legend=False, alpha=.2,
                      color=[cm[i] for i in clusters])
            #TODO ADD colored cluster centers as lines
            plt.figure() #FIXME: works only with weekdays
            day_clusters = pd.DataFrame({y: Load.resample('d').mean().index.weekday,
                                         'clusters': clusters,
                                         'val':1})
            x_labels = "Mon Tue Wed Thu Fri Sat Sun".split()
            day_clusters.pivot_table(columns=y, index='clusters',
                                     aggfunc='count').T.plot.bar(stacked=True)
            plt.gca().set_xticklabels(x_labels)
        except Exception: #FIXME: specify exception
            print ('Works only with daily profile clustering')

    return clusters_center.T


def get_load_stats(Load, per='a'):
    """Find load profile characteristics. Among other it estimates: peak, load factor, base load factor, operating hours,

    Arguments:
        Load: timeseries of load to be examined. A timeseries index is needed.
        per: reporting periods. Annual by default. Based on pandas time offsets
    Returns:
         dict: Parameter dictionary
    """
     #TODO 2D
    from .stats import all_stats_desc

    Load1 = clean_convert(Load, force_timed_index=True, freq='h')

    g = Load1.groupby(pd.Grouper(freq=per))
    if len(g) > 100:
        print ('Warning: Too many periods ({}) selected'.format(len(g)))
    p_dict = {}
    for period, load_per in g:
        ind = str(period.to_period())
        p_dict[ind] = {k: v(load_per) for k, v in all_stats_desc.items()}  #  named tuple instead of dict?
    return pd.DataFrame.from_dict(p_dict)


def _n_colors_from_colormap(n, cmap='Set1'):
    """ Returns lists of color tuples(RGBA)
        n: number of colours
        cmap: matplotlib colormap
    """
    from matplotlib.cm import get_cmap
    cm = get_cmap(cmap)
    return [cm(1.*i/n) for i in range(n)]

def detect_outliers(Load, threshold=None, window=5, plot_diagnostics=False):
    """ Detect and optionally remove outliers based on median rolling window filtering.
    Inspired by https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/

    Arguments:
        Load: input timeseries
        threshold: if None then 3 sigma is selected as threshold
        window: how many values to check
        plot_diagnostics: Plot diagnostics to check whether the outliers were removed accurately
    Return:
        index position of detected outliers
   """
    # TODO : Clean zero values (interpolate)

    a = Load.rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(Load - a)

    if threshold is None:
        threshold = 3 * np.std(Load)
    outlier_idx = difference > threshold
    if plot_diagnostics:
        if len(outlier_idx > 0):
            kw = dict(marker='o', linestyle='none', color='r', alpha=0.5)
            Load.plot()
            Load[outlier_idx].plot(**kw)
        else:
            print('No outliers detected. If you think that there are, try to raise the threshold')
    return outlier_idx


def countweekend_days_per_month(df, weekdays=True):     #TODO generalize count_x_per_y
    """Count number of occurrences where the day is Saturday or Sunday. Loops per month"""
    from collections import Counter
    out = []
    for __, values in df.index.groupby(df.index.month).items():
        m_count = Counter([day.weekday() for day in values])
        out.append(m_count[5] + m_count[6]) # 5,6 is saturday sunday
    return out