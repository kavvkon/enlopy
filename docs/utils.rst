.. _utils_module:

enlopy.utils: Utility Functions
==============================

The ``enlopy.utils`` module provides essential helper functions that support the
rest of the ``enlopy`` package. These utilities primarily focus on data
conversion, ensuring that timeseries data is in a consistent pandas format
with a ``DatetimeIndex``, which is crucial for most energy analysis tasks.

Core Functionalities
--------------------

*   **Timeseries Creation:** Standardizing the creation of pandas Series or DataFrames
    with a proper ``DatetimeIndex``.
*   **Data Cleaning and Conversion:** Robustly converting various input data types
    (lists, NumPy arrays, existing pandas objects) into a consistent timeseries format.

Rationale and Use Cases of Key Functions
----------------------------------------

Below is a description of the main utility functions and their purpose.
For detailed API parameters, please refer to the :ref:`API documentation <API>`.

.. contents:: Key Functions
   :local:
   :depth: 1

make_timeseries
~~~~~~~~~~~~~~~
*   **Rationale:** Many energy analyses require data to be indexed by time. This
    function provides a convenient way to create a pandas Series or DataFrame
    with a ``DatetimeIndex``, even from raw NumPy arrays or lists. It handles
    the generation of the time index based on specified start dates, lengths,
    and frequencies. It includes intelligent defaults for frequency if the input
    data length matches common patterns (e.g., 8760 for hourly annual data).
*   **Use Case:** Converting a simple list or NumPy array of 8760 hourly load values
    into a pandas Series with an hourly ``DatetimeIndex`` starting from January 1st
    of a specified year. Creating an empty timeseries structure with a defined
    frequency and length to be filled later.

clean_convert
~~~~~~~~~~~~~
*   **Rationale:** Functions within ``enlopy`` expect input data in a consistent
    format (typically a pandas Series or DataFrame with a ``DatetimeIndex``).
    This utility acts as a flexible and robust converter for various input types
    (Python lists, NumPy arrays, pandas Series without a proper index, etc.).
    It ensures that the data is in the correct pandas structure and can optionally
    force the creation of a ``DatetimeIndex`` using ``make_timeseries``.
*   **Use Case:** Internally, most ``enlopy`` functions use ``clean_convert`` at
    the beginning to preprocess input `Load` data. This makes the main functions
    more resilient to different data input types provided by the user, ensuring
    they can operate on a standardized timeseries representation. For example,
    if a user passes a NumPy array to a plotting function, ``clean_convert``
    would transform it into a pandas Series with a ``DatetimeIndex`` before plotting.

human_readable_time
~~~~~~~~~~~~~~~~~~~
*   **Rationale:** To convert a duration (e.g., a number of hours or seconds) into a more
    easily understandable string format, like "2 years 3 months 5 days".
*   **Use Case:** Displaying simulation lengths or time differences in reports or log
    messages in a format that is easier for humans to interpret than raw seconds or hours.
