Welcome to enlopy's documentation!
==================================
:Version: |version|
:Date: |today|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::
   :maxdepth: 3

   API

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
* ``Generate``: generate from daily and monthly profiles, generate from sinusoidal function, sample from given load duration curve, or from given PSD, add noise gaussian and autoregressive noise, generate correlated load profiles , fit to analytical load duration curve
* ``Statistics``: Feature extraction from timeseries for a quick overview of the characteristics of any load curve. Useful when coupled with machine learning packages.

This library is not focusing on regression and prediction (e.g. ARIMA, state-space etc.), since there are numerous relevant libraries around.

The documentation is under development. Please check the source code, the :ref:`API` or the example `jupyter notebook <https://github.com/kavvkon/enlopy/blob/master/notebooks/Basic%20examples.ipynb>`__ in the github repository for feature details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
