from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

import numpy as np

from .analysis import reshape_timeseries, clean_convert, get_LDC

__all__ = ['plot_heatmap', 'plot_3d', 'plot_percentiles', 'plot_rug', 'plot_boxplot', 'plot_LDC' ]

def plot_heatmap(Load, x='dayofyear', y='hour', aggfunc='sum', bins=8,
                figsize=(16,6), edgecolors='none', cmap='Oranges', colorbar=True, ax=None, **pltargs):
    """ Returns a 2D heatmap of the reshaped timeseries based on x, y
    Arguments:
        Load: 1D pandas with timed index
        x: Parameter for :meth:`enlopy.analysis.reshape_timeseries`
        y: Parameter for :meth:`enlopy.analysis.reshape_timeseries`
        bins: Number of bins for colormap
        edgecolors: colour of edges around individual squares. 'none' or 'w' is recommended.

        cmap: colormap name (from colorbrewer, matplotlib etc.)
        **pltargs: Exposes matplotlib.plot arguments
    Returns:
        2d heatmap
    """
    x_y = reshape_timeseries(Load, x=x, y=y, aggfunc=aggfunc)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cmap_obj = cm.get_cmap(cmap, bins)
    heatmap = ax.pcolor(x_y, cmap=cmap_obj, edgecolors=edgecolors, **pltargs)
    if colorbar:
        fig.colorbar(heatmap)
    ax.set_xlim(right=len(x_y.columns))
    ax.set_ylim(top=len(x_y.index))
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plot_3d(Load, x='dayofyear', y='hour', aggfunc='sum', bins=15,
           cmap='Oranges', colorbar=True, **pltargs):
    """ Returns a 3D plot of the reshaped timeseries based on x, y

    Arguments:
        Load: 1D pandas with timed index
        x: Parameter for :meth:`enlopy.analysis.reshape_timeseries`
        y: Parameter for :meth:`enlopy.analysis.reshape_timeseries`
        bins: Number of bins for colormap
        cmap: colormap name (from colorbrewer, matplotlib etc.)
        **pltargs: Exposes :meth:`matplotlib.pyplot.surface` arguments
    Returns:
        3d plot
    """
    import mpl_toolkits.mplot3d  # necessary for orojection=3d

    x_y = reshape_timeseries(Load, x=x, y=y, aggfunc=aggfunc)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    cmap_obj = cm.get_cmap(cmap, bins)
    X, Y = np.meshgrid(range(len(x_y.columns)), range(len(x_y.index)))
    surf = ax.plot_surface(X, Y, x_y, cmap=cmap_obj, rstride=1, cstride=1,
                           shade=False, antialiased=True, lw=0, **pltargs)
    if colorbar:
        fig.colorbar(surf)
    # Set viewpoint.
    # ax.azim = -130
    ax.elev = 45
    ax.auto_scale_xyz([0, len(x_y.columns)],
                      [0, len(x_y.index)],
                      [0, x_y.max().max()])
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plot_percentiles(Load, x='hour', zz='week', perc_list=[[5, 95], [25, 75], 50], ax=None, color='blue', **kwargs):
    """Plot predefined percentiles per timestep

    Arguments:
        Load: 1D pandas with timed index
        x (str): x axis aggregator. See :meth:`enlopy.analysis.reshape_timeseries`
        zz (str): similar to above for y axis
        perc_list(list): List of percentiles to plot. If it is an integer then it will be plotted as a line. If it is list it has to contain two items and it will be plotted using fill_between()
        **kwargs: exposes arguments of :meth:`matplotlib.pyplot.fill_between`
    Returns:
        Plot

    """
    if ax is None: # Hack for nice jupyter notebook compatibility
        ax=plt.gca()
    a = reshape_timeseries(Load, x=x, y=zz, aggfunc='mean')
    xx = a.columns.values

    # TODO: s 2s 3s instead of percentiles

    for i in perc_list:
        if len(np.atleast_1d(i)) == 1:
            perc = a.apply(lambda x: np.nanpercentile(x.values, i), axis=0)
            ax.plot(xx, perc.values, color='black')
        elif len(np.atleast_1d(i)) == 2:
            perc0 = a.apply(lambda x: np.nanpercentile(x.values, i[0]), axis=0)
            perc1 = a.apply(lambda x: np.nanpercentile(x.values, i[1]), axis=0)

            ax.fill_between(xx, perc0, perc1, lw=.5, alpha=.3, color=color, **kwargs)
        else:
            raise ValueError('List items should be scalars or 2-item lists')

    ax.set_xlim(left=min(xx), right=max(xx))
    ax.set_xlabel(x)



def plot_boxplot(Load, by='day', **pltargs):
    """Return boxplot plot for each day of the week

    Arguments:
        Load (pd.Series): 1D pandas Series with timed index
        by (str): group results by 'day' or 'hour'
        **pltargs (dict): Exposes :meth:`matplotlib.pyplot.plot` arguments
    Returns:
        plot
    """
    Load = clean_convert(Load,force_timed_index=True)

    if by == 'day':
        grp = Load.groupby(Load.index.weekday)
        labels = "Mon Tue Wed Thu Fri Sat Sun".split()
    elif by == 'hour':
        grp = Load.groupby(Load.index.hour)
        labels = np.arange(0, 24)
    else:
        raise NotImplementedError('Only "day" and "hour" are implemented')
    a = []
    for __, value in grp:
        a.append(value)
    plt.boxplot(a, labels=labels, **pltargs)
    # TODO : Generalize to return monthly, hourly etc.
    # TODO Is it really needed? pd.boxplot()


def plot_LDC(Load, stacked=True, x_norm=True, y_norm=False, cmap='Spectral', color='black',
             legend=False, zoom_peak=False, ax=None, **kwargs):
    """Plot Load duration curve
    
    Arguments:
        Load (pd.Series): 1D pandas Series with timed index
        x_norm (bool): Normalize x axis (0,1)
        y_norm (bool): Normalize y axis (0,1)
        color (str): color of line. For Series only (1D)
        cmap (str): Colormap of area. For Dataframes only (2D)
        legend (bool): Show legend. For Dataframes only (2D)
        zoom_peak (bool): Show zoomed plot of peak
        kwargs (dict): exposes arguments of pd.DataFrame.plot.area
    Returns:
        Load duration curve plot
    """
    if ax is None:
        __ = plt.figure(1)
        ax_main = plt.axes()
    else:
        ax_main = ax

    if Load.ndim >= 2:
        if stacked:
            x, y = get_LDC(Load, x_norm=x_norm, y_norm=y_norm)
            # Reconverting to Dataframe as pd.plot.area is much more robust than plt.stackplot

            ldc_frame = pd.DataFrame(y, index=x, columns=Load.columns)
            y_max = np.nanmax(np.nansum(y, axis=1))
            if y_norm: # We need to renormalize cause the sum of columns is more than 1.0
                ldc_frame = ldc_frame/ y_max
            ldc_frame.plot.area(cmap=cmap, lw=0, legend=legend, ax=ax_main, **kwargs)
            y_max = np.nanmax(np.nansum(y, axis=1))
        else:
            for __, v in Load.items():
                x, y = get_LDC(v, x_norm=x_norm, y_norm=y_norm)
                ax_main.plot(x, y, color=color)
            y_max = np.nanmax(y)
    else:
        x, y = get_LDC(Load, x_norm=x_norm, y_norm=y_norm)
        ax_main.plot(x, y, color=color)
        y_max = np.nanmax(y)

    # Set axes labels
    ax_x_min = np.min(x)
    if x_norm:
        ax_x_max = 1
        xlabel = 'Normalized duration'
    else:
        ax_x_max = len(y)
        xlabel = 'Duration'
    if y_norm:
        ax_y_max = 1
        ylabel = 'Normalized Power'
    else:
        ax_y_max = y_max
        ylabel = 'Power'

    ax_main.set_xlim(ax_x_min, ax_x_max)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylim(0, ax_y_max * 1.01)
    ax_main.set_ylabel(ylabel)
    # Draw inset plot
    if zoom_peak:
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        axins = zoomed_inset_axes(ax_main, 2.5, loc=1)
        if y.ndim >= 2:
            ldc_frame.plot.area(cmap=cmap, lw=0, legend=False, ax=axins, **kwargs)
        else:
            axins.plot(x, y, color=color)
        axins.set_xlim([ax_x_min, 0.15 * ax_x_max]) #TODO: Estimate x axis limits based on plotted values
        axins.set_ylim([0.8 * ax_y_max, ax_y_max])
        axins.get_xaxis().set_ticks([])
        axins.get_yaxis().set_ticks([])
        mark_inset(ax_main, axins, loc1=1, loc2=3, fc="none", ec="0.5")


def plot_rug(df_series, on_off=False, cmap='Greys', fig_title='', fig_width=14, normalized=False):
    """Create multiaxis rug plot from pandas Dataframe
    
    Arguments:
        df_series (pd.DataFrame): 2D pandas with timed index
        on_off (bool): if True all points that are above 0 will be plotted as one color. If False all values will be colored based on their value.
        cmap (str): colormap name (from colorbrewer, matplotlib etc.)
        fig_title (str): Figure title
        normalized (bool): if True, all series colormaps will be normalized based on the maximum value of the dataframe
    Returns:
        plot
    """
    def format_axis(iax):
        # Formatting: remove all lines (not so elegant)
        for spine in ['top','right','left','bottom']:
            iax.axes.spines[spine].set_visible(False)
            #iax.spines['right'].set_visible(False)

        # iax.xaxis.set_ticks_position('none')
        iax.yaxis.set_ticks_position('none')
        iax.get_yaxis().set_ticks([])
        iax.yaxis.set_label_coords(-.05, -.1)

    def flag_operation(v):
        if np.isnan(v) or v == 0:
            return False
        else:
            return True

    # check if Series or dataframe
    if isinstance(df_series, pd.DataFrame):
        rows = len(df_series.columns)
    elif isinstance(df_series, pd.Series):
        df_series = df_series.to_frame()
        rows = 1
    else:
        raise ValueError("Has to be either Series or Dataframe")
    if len(df_series) < 1:
        raise ValueError("Has to be non empty Series or Dataframe")

    max_color = np.nanmax(df_series.values)
    min_color = np.nanmin(df_series.values)

    __, axes = plt.subplots(nrows=rows, ncols=1, sharex=True,
                             figsize=(fig_width, 0.25 * rows), squeeze=False,
                             frameon=False, gridspec_kw={'hspace': 0.15})

    for (item, iseries), iax in zip(df_series.iteritems(), axes.ravel()):
        format_axis(iax)
        iax.set_ylabel(str(item)[:30], rotation='horizontal',
                       rotation_mode='anchor',
                       horizontalalignment='right', x=-0.01)
        x = iseries.index

        if iseries.sum() > 0:  # if series is not empty
            if on_off:
                i_on_off = iseries.apply(flag_operation).replace(False, np.nan)
                i_on_off.plot(ax=iax, style='|', lw=.7, cmap=cmap)
            else:
                y = np.ones(len(iseries))
                #Define (truncated) colormap:
                if not normalized:  # Replace max_color (frame) with series max
                    max_color = np.nanmax(iseries.values)
                    min_color = np.nanmin(iseries.values)
                # Hack to plot max color when all series are equal
                if np.isclose(min_color, max_color):
                    min_color = min_color * 0.99

                iax.scatter(x, y,
                            marker='|', s=100,
                            c=iseries.values,
                            vmin=min_color,
                            vmax=max_color,
                            cmap=cmap)

    axes.ravel()[0].set_title(fig_title)
    axes.ravel()[-1].spines['bottom'].set_visible(True)
    axes.ravel()[-1].set_xlim(np.min(x), np.max(x))


def plot_line_holidays():
    #ax.vspan if day == holiday
    #should work only with daily
    pass

def describe_load(Load):
    """Summary plot that describes the most important features of the passed timeseries """
    pass