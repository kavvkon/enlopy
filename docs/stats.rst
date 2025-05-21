.. _stats_module:

enlopy.stats: Extracting Statistical Features from Timeseries
=============================================================

The ``enlopy.stats`` module provides a suite of functions for calculating
various statistical properties and extracting descriptive features from
timeseries data. These functions are valuable for characterizing load profiles,
understanding variability, and preparing data for machine learning applications.
Many of these functions are utilized by ``enlopy.analysis.get_load_stats``
to generate summary statistics.

Core Functionalities
--------------------

The module offers calculations for:

*   Basic descriptive statistics (mean, load factor, percentiles).
*   Trend and periodicity analysis.
*   Duration of specific conditions (e.g., zero load).
*   Ramp rate characterization.
*   Autocorrelation and peak detection.

`all_stats_desc` Dictionary
---------------------------

A key component of this module is the ``all_stats_desc`` dictionary.
This dictionary maps human-readable names of statistical features (e.g.,
'Load Factor (peakiness)', 'Total Zero load duration') to specific functions
(often partially applied versions of the standalone functions in this module).
This provides a convenient way to compute a standardized set of features,
as used by ``enlopy.analysis.get_load_stats``.

Rationale and Use Cases of Key Functions
----------------------------------------

Below is a description of some notable functions and concepts within the module.
For detailed API parameters of individual functions, please refer to the
:ref:`API documentation <API>`.

.. contents:: Key Functions and Concepts
   :local:
   :depth: 1

Basic Statistics (get_mean, get_lf, get_percentile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** These functions provide fundamental statistical measures.
    `get_mean` calculates the average value. `get_lf` (Load Factor) is crucial in
    energy analysis, representing the ratio of average load to peak load, indicating
    how efficiently capacity is utilized. `get_percentile` helps understand the
    distribution of values.
*   **Use Case:** Calculating the annual load factor of an electricity grid to assess
    overall system efficiency. Determining the 95th percentile of load to understand
    near-peak demand levels.

Trend and Periodicity (get_trend, get_highest_periodicity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** `get_trend` fits a linear trend to the data, helping to identify
    long-term increases or decreases. `get_highest_periodicity` uses spectral
    analysis (Welch's method) and peak finding to identify the dominant cycles
    or seasonalities present in the timeseries.
*   **Use Case:** Identifying if there's an increasing trend in annual energy consumption.
    Detecting daily, weekly, or annual cycles in a load profile.

Duration and Ramping (get_rle, largest_dur_of_zero, get_dur_val, get_ramp_rates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** These functions characterize how long certain conditions last and
    how quickly values change. `get_rle` (Run Length Encoding) is a general utility
    to count consecutive identical values. `largest_dur_of_zero` and `get_dur_val`
    focus on periods of zero or specific values, important for understanding
    downtime or baseload. `get_ramp_rates` measures the speed of load increase/decrease,
    critical for assessing grid flexibility needs.
*   **Use Case:** Determining the longest continuous period a generator was offline
    (zero output). Calculating the maximum rate at which solar power output ramps up
    on a clear morning.

Other Characteristics (get_peaks, get_load_ratio, get_autocorr)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** `get_peaks` identifies significant peaks in the data. `get_load_ratio`
    (max/min) gives a simple measure of variability. `get_autocorr` measures how much
    a timeseries is correlated with a lagged version of itself, indicating persistence
    or repetitiveness.
*   **Use Case:** Finding the times of daily peak demand in an electricity load profile.
    Assessing if a high load value today implies a higher likelihood of a high load
    value tomorrow (autocorrelation).

Using `all_stats_desc`
~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** The ``all_stats_desc`` dictionary provides a predefined collection
    of these statistical measures, making it easy to compute a comprehensive profile
    of a timeseries. Each entry pairs a descriptive string with a function from this
    module (sometimes with specific parameters preset using ``functools.partial``).
*   **Use Case:** This dictionary is directly used by ``enlopy.analysis.get_load_stats``
    to generate a DataFrame of various load characteristics for different time periods
    (e.g., for each month in a year). Users can also iterate through this dictionary
    to apply a standard set of statistical analyses to their own timeseries data.
