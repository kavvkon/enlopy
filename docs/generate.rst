.. _generate_module:

enlopy.generate: Generating Energy Timeseries
=============================================

The ``enlopy.generate`` module provides a suite of tools for creating, synthesizing,
and manipulating energy-related timeseries data. These functions are essential
for simulations, modeling alternative scenarios, data augmentation, or when
actual high-resolution data is unavailable.

Core Functionalities
--------------------

The module covers several aspects of timeseries generation:

*   **Creating profiles from base data:** Generating higher-resolution series from coarser data (e.g., daily to hourly) or from typical profiles.
*   **Stochastic modeling:** Creating realistic synthetic timeseries based on statistical properties.
*   **Transformations:** Modifying existing timeseries by adding noise, simulating demand response, or removing outliers.
*   **Specialized generation:** Creating loads from Load Duration Curves (LDCs) or Power Spectral Densities (PSDs).

Rationale and Use Cases of Key Functions
----------------------------------------

Below is a description of some key functions, their purpose, and typical use cases.
For detailed API parameters, please refer to the :ref:`API documentation <API>`.

.. contents:: Key Functions
   :local:
   :depth: 1

disag_upsample
~~~~~~~~~~~~~~
*   **Rationale:** Often, energy data is available at a coarse granularity (e.g., daily consumption),
    but models or analyses require higher resolution (e.g., hourly). This function
    distributes the coarser data points into finer intervals based on a representative
    disaggregation profile, ensuring the total sum over the original period is preserved.
*   **Use Case:** Converting daily household energy consumption data to hourly data using a
    standard hourly consumption profile for that type of household.

gen_daily_stoch_el
~~~~~~~~~~~~~~~~~~
*   **Rationale:** To create realistic, synthetic daily electricity load profiles when only
    aggregate daily energy is known or when multiple variations are needed for robust analysis.
    It uses pre-defined statistical means and standard deviations (derived from analysis
    of many households) per timestep, combined with a Gauss-Markov process to introduce
    autocorrelation.
*   **Use Case:** Generating diverse daily load profiles for a set of simulated households
    in an agent-based model, where each household has a total daily energy consumption target.

gen_load_from_daily_monthly
~~~~~~~~~~~~~~~~~~~~~~~~~~~
*   **Rationale:** Constructing an annual hourly load profile when only monthly total consumption
    and typical daily profiles (for weekdays and weekends) are available. This is common
    in energy planning or when detailed historical data is scarce.
*   **Use Case:** Creating a year-long hourly electricity demand forecast for a region
    based on projected monthly energy demands and established daily usage patterns for
    residential and commercial sectors.

gen_load_sinus
~~~~~~~~~~~~~~
*   **Rationale:** To generate synthetic timeseries that exhibit clear periodic behavior
    at multiple timescales (daily, weekly, annually). This is useful for creating
    baseline profiles or test data for models that need to capture seasonality.
*   **Use Case:** Creating a synthetic temperature profile or a baseline renewable energy
    generation profile that follows predictable daily and annual cycles.

gen_corr_arrays
~~~~~~~~~~~~~~~
*   **Rationale:** In many energy systems, multiple variables are correlated (e.g., wind
    speed and solar irradiance at different locations, or electricity prices and demand).
    This function generates multiple arrays of random numbers that exhibit a specified
    correlation structure, essential for Monte Carlo simulations or for generating
    realistic multi-variate inputs.
*   **Use Case:** Generating correlated wind speed timeseries for several nearby wind farms
    to assess the aggregated power output variability.

gen_load_from_LDC
~~~~~~~~~~~~~~~~~
*   **Rationale:** To create a sequence of load values that statistically matches a given
    Load Duration Curve (LDC). The LDC represents the amount of time the load is at or
    above a certain level. This method uses inverse transform sampling.
*   **Important Note:** This method generates values that match the LDC's distribution
    but **loses the original temporal sequence**. The output is a set of load values,
    not a chronologically realistic timeseries. It's often a precursor to `gen_load_from_PSD`.
*   **Use Case:** Generating a set of hourly load values for a year that, when sorted,
    will precisely match a target LDC for planning purposes.

gen_load_from_PSD
~~~~~~~~~~~~~~~~~
*   **Rationale:** To generate a realistic timeseries that not only matches a target
    probability distribution (often derived from an LDC via `gen_load_from_LDC`)
    but also possesses specific spectral characteristics (i.e., how power is distributed
    across different frequencies, indicating temporal patterns like ramps, cycles).
    It uses the Iterated Amplitude Adjusted Fourier Transform (IAAFT) algorithm.
*   **Use Case:** Taking hourly load values generated by `gen_load_from_LDC` and
    "shuffling" them to create a chronologically realistic annual load profile that
    exhibits typical daily and weekly patterns (captured in the PSD).

gen_gauss_markov
~~~~~~~~~~~~~~~~
*   **Rationale:** To generate timeseries that exhibit autoregressive properties, meaning
    future values depend on past values, along with some randomness. This is useful for
    modeling systems with inertia or memory, where values don't change erratically
    but smoothly transition.
*   **Use Case:** Simulating short-term load fluctuations or temperature variations where
    the current value is strongly influenced by the immediately preceding values.

add_noise
~~~~~~~~~
*   **Rationale:** To introduce variability or uncertainty into an existing timeseries.
    Real-world data is rarely perfectly smooth, and adding noise can make simulations
    more realistic or test the robustness of models.
*   **Use Case:** Adding random fluctuations to a deterministic solar power generation
    profile to account for unpredictable cloud cover.

gen_analytical_LDC
~~~~~~~~~~~~~~~~~~
*   **Rationale:** To quickly generate a standard Load Duration Curve shape based on
    a few key empirical parameters (Peak load, capacity factor, base load factor,
    operating hours). This avoids needing full timeseries data to get an LDC.
*   **Use Case:** Quickly sketching an LDC for a system where only high-level statistics
    are known, for initial capacity planning or policy analysis.

gen_demand_response
~~~~~~~~~~~~~~~~~~~
*   **Rationale:** To simulate the impact of demand response programs, which aim to
    reduce peak loads by either shifting demand to off-peak hours or by curtailing
    (shaving) load during peak times.
*   **Use Case:** Assessing how much a utility can reduce its peak capacity requirements
    by implementing a residential demand response program that shifts a certain percentage
    of peak load.

remove_outliers
~~~~~~~~~~~~~~~
*   **Rationale:** Outliers in timeseries data can distort analysis and modeling. This
    function first detects outliers (using methods from `enlopy.analysis`) and then
    replaces them with interpolated values, providing a cleaner dataset.
*   **Use Case:** Preprocessing a measured electricity demand timeseries to remove anomalous
    readings caused by sensor errors before using it for forecasting.
