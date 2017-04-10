import numpy as np
import pandas as pd

from enload.utils import make_timeseries
from enload.analysis import reshape_timeseries


def test_reshape_timeseries():
    a = np.random.rand(8760)
    b = reshape_timeseries(a, x='dayofyear', y='hour')
    assert b.shape == (24,365)
    assert np.isclose(b.sum().sum(), a.sum())

def test_reshape_multiannual():
    a = np.random.rand(8760*2)
    a = make_timeseries(a, freq='h')
    b = reshape_timeseries(a, x='dayofyear', y='hour', aggfunc='sum')
    assert b.shape == (24,365)
    assert np.isclose(b.sum().sum(), a.sum())



