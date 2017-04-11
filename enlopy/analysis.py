import numpy as np
import pandas as pd

from .utils import clean_convert, _freq_to_sec

__all__ = ['reshape_timeseries', 'get_LDC', 'get_load_archetypes', 'get_load_stats']

def reshape_timeseries(Load, x='dayofyear', y=None, aggfunc='sum'):
    """Returns a reshaped pandas DataFrame that shows the aggregated load for selected
    timeslices. e.g. time of day vs day of year

    Parameters:
        Load (pd.Series): timeseries
        x (str): x axis aggregator. Has to be an accessor of pd.DatetimeIndex
         (year, dayoftime, week etc.)
        y (str): similar to above for y axis
    Returns:
        reshaped pandas dataframe according to x,y
    """

    # Have to convert to dataframe in order for pivottable to work
    # 1D, Dataframe
    a = clean_convert(Load, force_timed_index=True, always_df=True)
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


def get_LDC(Load, x_norm=True, y_norm=False, bins=999, trunc_0=False):
    """Generates the Load Duration Curve based on a given load
    
    Arguments:
        Load (pd.Series): timeseries
        x_norm (bool): Normalize x axis (0,1)
        y_norm (bool): Normalize y axis (0,1)
        bins (int): how many values to plot
        trunc_0 (bool): If true remove all values under zero
    Returns:
        np.ndarray: array [x, y] ready for plotting (e.g. plt(\*LDC_load(load)))
    """

    # remove nan because histogram does not work
    load_masked = Load[~np.isnan(Load)]
    n, xbins = np.histogram(load_masked, bins=bins, density=True)
    # xbins = xbins[:-1] #remove last element to make equal size
    cum_values = np.zeros(xbins.shape)
    cum_values[1:] = np.cumsum(n*np.diff(xbins))
    out = np.array([1-cum_values, xbins])
    # out = np._r[[1 0], out] # Add extra point
    if trunc_0: # Trunc non zero elements
        out[out < 0] = 0
    return out


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
    clusters_center, dist = kmeans(df_white, k)

    if plot_diagnostics:
        try:
            import matplotlib.pyplot as plt
            clusters, _ = vq(df_white, clusters_center)
            cm = _n_colors_from_colormap(k)
            df.T.plot(legend=False, alpha=.2,
                      color=[cm[i] for i in clusters])
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
        load: timeseries of load to be examined
        per: reporting periods. Annual by default. Based on pandas time offsets 
    Returns:
         dict: Parameter dictionary
    """
     #TODO 2D
    from .stats import all_stats_desc
    g = Load.groupby(pd.TimeGrouper(per))
    if len(g) > 100:
        print ('Waning: {} periods selected'.format(len(g)))
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


def countweekend_days_per_month(df, weekdays=True):     #TODO generalize count_x_per_y
    """Count number of occurrences where the day is Saturday or Sunday. Loops per month"""
    from collections import Counter
    out = []
    for month, values in df.index.groupby(df.index.month).items():
        m_count = Counter([day.weekday() for day in values])
        out.append(m_count[5] + m_count[6]) # 5,6 is saturday sunday
    return out
    #TODO fix to daily