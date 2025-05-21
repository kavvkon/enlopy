.. _plot_module:

enlopy.plot: Visualizing Energy Timeseries
===========================================

The ``enlopy.plot`` module provides a collection of functions for visualizing
energy-related timeseries data. These plotting utilities are designed to reveal
patterns, trends, and distributions within the data, often working in conjunction
with transformations from the ``enlopy.analysis`` module.

Core Visualizations
-------------------

The module offers several types of plots common in energy analysis:

*   **Heatmaps and 3D plots:** For visualizing load across two time dimensions.
*   **Percentile plots:** To understand temporal variations in load distribution.
*   **Boxplots:** To compare distributions across different time categories.
*   **Load Duration Curve (LDC) plots:** Standard visualization for power system analysis.
*   **Rug plots:** For displaying activity or comparing multiple timeseries.

Rationale and Use Cases of Key Functions
----------------------------------------

Below is a description of key plotting functions, their purpose, and typical use cases.
For detailed API parameters, please refer to the :ref:`API documentation <API>`.

.. contents:: Key Functions
   :local:
   :depth: 1

plot_heatmap
~~~~~~~~~~~~
*   **Rationale:** Heatmaps are an effective way to visualize the magnitude of a variable
    across two dimensions. For timeseries, this typically involves reshaping the data
    (e.g., using ``enlopy.analysis.reshape_timeseries``) so that one time attribute
    (like hour of day) forms one axis, and another (like day of year) forms the other.
    Color intensity represents the load magnitude.
*   **Use Case:** Visualizing an entire year's hourly electricity demand to quickly
    identify periods of high/low consumption, seasonal trends, and daily patterns.
    For example, seeing bright colors during summer afternoons (AC load) and winter
    evenings (heating/lighting).

plot_3d
~~~~~~~
*   **Rationale:** Similar to heatmaps, 3D surface plots can represent load magnitude
    across two time dimensions, but with the load value explicitly shown on the Z-axis.
    This can sometimes offer a more intuitive grasp of peaks and valleys in the data.
*   **Use Case:** Creating a 3D representation of hourly load versus day of year to
    emphasize the height of peak demand periods and the depth of low-demand troughs.

plot_percentiles
~~~~~~~~~~~~~~~~
*   **Rationale:** To understand how the distribution of load values changes over a
    specific cycle (e.g., daily, weekly). This function plots user-defined percentiles
    (e.g., 5th, 25th, 50th (median), 75th, 95th) for each point in the cycle,
    showing the typical range and variability of the load.
*   **Use Case:** Plotting hourly percentiles of electricity demand for each day of the
    week. This can show, for instance, that while median load on weekends is lower,
    the variability (spread between 5th and 95th percentiles) might be higher or different
    in shape compared to weekdays.

plot_rug
~~~~~~~~
*   **Rationale:** Rug plots are useful for visualizing the activity or values of multiple
    timeseries simultaneously in a compact way. Each timeseries is represented by a
    horizontal "rug." For on/off data, dashes can indicate "on" periods. For continuous
    data, the color or intensity of dashes can represent magnitude.
*   **Use Case:** Displaying the operational status (on/off) of multiple appliances in a
    household over a day. Or, visualizing the normalized output of several renewable
    energy sources (wind, solar) over time to see their collective behavior.

plot_boxplot
~~~~~~~~~~~~
*   **Rationale:** Boxplots (or box-and-whisker plots) provide a standardized way to
    display the distribution of data based on a five-number summary (minimum, first
    quartile, median, third quartile, maximum). They are excellent for comparing
    distributions across different categories.
*   **Use Case:** Comparing the distribution of hourly electricity demand for each day
    of the week. This can clearly show differences in median load, variability (interquartile
    range), and the presence of outliers for weekdays versus weekend days.

plot_LDC
~~~~~~~~
*   **Rationale:** Visualizing the Load Duration Curve (LDC), which is typically generated
    by ``enlopy.analysis.get_LDC``. This plot shows the relationship between load levels
    and the duration for which those levels are met or exceeded. It's a standard tool for
    assessing power system adequacy and operational characteristics.
*   **Use Case:** Plotting the LDC for a regional electricity system to visualize how many
    hours per year different levels of generation capacity are utilized. Options allow
    for plotting multiple LDCs (e.g., for different scenarios or sub-regions) and
    zooming into the peak portion of the curve.
