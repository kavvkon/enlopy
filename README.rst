Python toolkit for energy load time series
==========================================

|pyversion| |license| |version_status| |build_status| |docs|

``enlopy`` is an open source python library with methods to generate,
process, analyze, and plot energy related timeseries.

While it can be used for any kind of data it has a strong focus on those
that are related with energy i.e. electricity/heat demand or generation,
prices etc. The methods included here are carefully selected to
fit in that context and they had been, gathered, generalized and
encapsulated during the last years while working on different research
studies.

The aim is to provide a higher level API than the one that is already
available in commonly used scientific packages (pandas, numpy, scipy).
This facilitates the analysis and processing of energy load timeseries
that can be used for modelling and statistical analysis. In some cases it
is just a convenience wrapper of common packages just as pandas and in
other cases it implements methods or statistical models found in
literature.

It consists of four modules that include among others the following:

* ``Analysis``: Overview of descriptive statistics, reshape, load duration curve, extract daily archetypes (clustering)
* ``Plot``: 2d heatmap, 3d plot, boxplot, rugplot
* ``Generate``: generate from daily and monthly profiles, generate from sinusoidal function, sample from given load duration curve, or from given PSD, add noise gaussian and autoregressive noise, genrate correlated load profiles , fit to analytical load duration curve
* ``Statistics``: Feature extraction from timeseries for a quick overview of the characteristics of any load curve. Useful when coupled with machine learning packages.

This library is not focusing on regression and prediction (e.g. ARIMA, state-space etc.), since there are numerous relevant libraries around.

Example
-------
Try to run one of the following commands to explore some of this library's features:

.. code:: python

    >>> import numpy as np
    >>> import enlopy as el
    >>> Load = np.random.rand(8760) # Create random vector of values
    >>> eload = el.make_timeseries(Load) # Convenience wrapper around pandas timeseries

    >>> el.plot_heatmap(eload, x='day', y='month', aggfunc='mean') # Plots 2d heatmap
    >>> el.plot_percentiles(eload) # Plots mean and quantiles
    >>> el.plot_LDC(eload) # Plots a Load Duration Curve
    >>> el.plot_rug(eload) # Plots a nice rugplot. Useful for dispatch results
    >>> el.get_load_archetypes(eload, plot_diagnostics=True) # Splits daily loads in clusters (archetypes)

More examples can be found in `this jupyter notebook <https://github.com/kavvkon/enlopy/blob/master/notebooks/Basic%20examples.ipynb>`__.

Documentation
-------------
Detailed documentation is still under construction, but you can find an overview of the available methods here: http://enlopy.readthedocs.io/

Install
-------

Currently you can find the latest stable version in pypi. You can install it with:

::

    pip install enlopy

Be aware that this library is still in conceptual mode, so the API is most probably going to change in the following versions.
If you already have it installed and you want to upgrade to the latest stable version please use the following:

::

    pip install -U --upgrade-strategy only-if-needed enlopy

This will ensure that the dependencies will not be updated if the minimum requirements are already satisfied with the current version.

If you want to download the latest version from git for use or development purposes:

.. code:: bash

    git clone https://github.com/kavvkon/enlopy.git
    cd enlopy
    conda env create  # Automatically creates environment based on environment.yml
    source activate enlopy
    pip install -e . # Install editable local version

It should be ready to run out of the box for anyone that has the
`anaconda distribution <https://www.continuum.io/downloads>`__
installed. The only dependencies required to use ``enlopy`` are the
following:

-  `numpy <http://numpy.org>`__
-  `scipy <http://scipy.org>`__
-  `pandas <http://pandas.pydata.org/>`__
-  `matpotlib <http://matplotlib.org/>`__

Contribute
----------

If you think you can contribute with new relevant methods that you are
currently using or improve the code or documentation in any way, feel free to contact me,
fork the repository and send your pull requests.

Citing
------

If you use this library in an academic work, please consider citing it.

[1] K. Kavvadias, “enlopy: Python toolkit for energy load time series”

``enlopy`` has been already used for processing demand timeseries in this scientific paper:
http://dx.doi.org/10.1016/j.apenergy.2016.08.077

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/enlopy.svg
    :alt: Supported Python versions.
    :target: http://pypi.python.org/pypi/enlopy
.. |license| image:: https://img.shields.io/pypi/l/enlopy.svg
    :alt: BSD License
    :target: https://opensource.org/licenses/BSD-3-Clause
.. |version_status| image:: http://img.shields.io/pypi/v/enlopy.svg?style=flat
   :target: https://pypi.python.org/pypi/enlopy
.. |build_status| image:: http://img.shields.io/travis/kavvkon/enlopy/master.svg?style=flat
   :target: https://travis-ci.org/kavvkon/enlopy
.. |docs| image:: https://readthedocs.org/projects/enlopy/badge/
    :alt: Documentation
    :target: http://enlopy.readthedocs.io/en/latest/

