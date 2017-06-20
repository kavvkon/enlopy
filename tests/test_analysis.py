import numpy as np

from enlopy.utils import make_timeseries
from enlopy.analysis import reshape_timeseries, get_LDC, get_load_stats


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

def test_get_LDC():
    a = np.random.rand(8760*2)
    a = make_timeseries(a, freq='h')
    b = get_LDC(a)
    assert np.isclose(b[1].sum(), a.sum())
    #check monotonicity
    assert np.all(np.diff(b[1]) < 0)

def test_get_LDC_not_annual():
    a = np.random.rand(10000)
    a = make_timeseries(a, freq='h')
    b = get_LDC(a)
    assert np.isclose(b[1].sum(), a.sum())
    #check monotonicity
    assert np.all(np.diff(b[1]) < 0)

def test_get_LDC_2d():
    a = np.random.rand(8760, 4)
    a = make_timeseries(a, freq='h')
    b = get_LDC(a)
    assert np.isclose(np.nansum(b[1]), np.nansum(a))
    # check monotonicity
    assert np.all(np.diff(b[1].sum(1)) < 0)

def test_get_stats():
    a = np.ones(8760)
    a = make_timeseries(a, freq='h')
    b = get_load_stats(a)
    assert a.sum() == b.loc['Sum'].squeeze()
    assert np.asarray(b.loc['Ramps (98%)'])[0] == (0,0)
    assert np.isclose(0, b.loc['Trend'].squeeze())
    assert 1 == b.loc['Load Factor (peakiness)'].squeeze()
