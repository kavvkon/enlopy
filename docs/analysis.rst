.. _analysis_module:

enlopy.analysis: Analyzing Energy Timeseries
=============================================

The ``enlopy.analysis`` module offers a collection of functions designed to
inspect, characterize, and extract meaningful insights from energy-related
timeseries data. These tools are fundamental for understanding load patterns,
variability, and for preparing data for further modeling or reporting.

Core Functionalities
--------------------

The module focuses on:

*   **Data Transformation:** Reshaping timeseries for easier analysis and visualization.
*   **Load Characterization:** Calculating standard metrics like Load Duration Curves and key statistics.
*   **Pattern Recognition:** Identifying typical load profiles (archetypes) using clustering.
*   **Data Cleaning:** Detecting outliers.

Rationale and Use Cases of Key Functions
----------------------------------------

Below is a description of key functions, their purpose, and typical use cases.
For detailed API parameters, please refer to the :ref:`API documentation <API>`.

.. contents:: Key Functions
   :local:
   :depth: 1

reshape_timeseries
~~~~~~~~~~~~~~~~~~
*   **Rationale:** Timeseries data is often a long 1D array. Reshaping it into a 2D
    matrix based on time attributes (e.g., rows as hours of the day, columns as
    days of the year) allows for powerful visualizations (like heatmaps) and
    makes it easier to observe daily, weekly, or seasonal patterns.
*   **Use Case:** Transforming an annual hourly electricity demand series into a
    24 (hour) x 365 (day) matrix to visualize daily load shapes across the year
    using a heatmap. This can help identify when peak loads occur or how profiles
    change seasonally.

get_LDC
~~~~~~~
*   **Rationale:** The Load Duration Curve (LDC) is a fundamental tool in power system
    analysis. It sorts load values from highest to lowest, showing the percentage
    of time the load meets or exceeds a particular level. This helps in
    understanding the utilization of generation capacity and planning new investments.
*   **Use Case:** Analyzing an annual hourly load profile to determine for how many
    hours the system load is above 80% of its peak, which informs decisions about
    peaking power plant requirements. It can also be used to compare the "peakiness"
    of different load profiles.

get_load_archetypes
~~~~~~~~~~~~~~~~~~~
*   **Rationale:** In a large dataset of individual load profiles (e.g., from many
    smart meters), there are often recurring typical daily or weekly patterns.
    This function uses k-means clustering to identify these "archetypes" or
    representative profiles.
*   **Use Case:** Segmenting a population of residential electricity consumers based
    on their typical daily usage patterns (e.g., "night owls," "morning peaks,"
    "daytime constant") for targeted demand-side management programs or tariff design.

get_load_stats
~~~~~~~~~~~~~~
*   **Rationale:** To quickly summarize key characteristics of a load profile over
    defined periods (e.g., monthly, annually). This function computes metrics like
    peak load, average load, load factor (average/peak), base load factor, and
    total operating hours, providing a snapshot of the load's behavior. It leverages
    descriptors from the ``enlopy.stats`` module.
*   **Use Case:** Calculating monthly peak demand, average demand, and load factor for
    an industrial facility to track energy efficiency improvements or to report
    to energy regulators.

detect_outliers
~~~~~~~~~~~~~~~
*   **Rationale:** Anomalous data points (outliers) can skew statistical analyses
    and lead to incorrect conclusions or model behavior. This function provides a
    method to identify such outliers based on deviations from a rolling median,
    which is robust to the presence of outliers itself.
*   **Use Case:** Cleaning a timeseries of sensor data (e.g., temperature, power output)
    by identifying and flagging readings that are likely errors before further
    processing or analysis. The identified outliers can then be removed or imputed
    using ``enlopy.generate.remove_outliers``.

countweekend_days_per_month
~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** A utility function that counts the number of weekend days (Saturdays and Sundays)
    within each month of a given timeseries' DatetimeIndex. This can be useful for analyses
    that need to normalize or compare data based on the number of working vs. non-working days.
*   **Use Case:** Normalizing monthly energy consumption data by the number of business days in
    each month to get a more comparable measure of consumption intensity, especially when
    comparing different months or years.
