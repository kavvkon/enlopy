import numpy as np
import pandas as pd
from enlopy.utils import make_timeseries
from enlopy.generate import (add_noise, gen_daily_stoch_el, gen_load_from_daily_monthly, gen_load_sinus, gen_demand_response,
                             disag_upsample, clean_convert, countweekend_days_per_month,
                             gen_analytical_LDC, gen_load_from_LDC, gen_corr_arrays, gen_gauss_markov)

class Test_noise():

    def test_ndarray_gauss(self):
        a = np.random.rand(24)
        b = np.random.rand(24) / 10
        c = gen_gauss_markov(a, b, .9)
        assert isinstance(c, np.ndarray)

    def test_ndarray_add_noise_gauss(self):
        a = np.random.rand(8760)
        b = add_noise(a, 3, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.Series)

    def test_2d_ndarray_add_noise_gauss(self):
        a = np.random.rand(8760, 2)
        b = add_noise(a, 3, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.DataFrame)
        assert (8760,2) == b.shape

    def test_ndarray_add_noise_normal(self):
        a = np.random.rand(8760)
        b = add_noise(a, 1, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.Series)

    def test_2d_ndarray_add_noise_normal(self):
        a = np.random.rand(8760, 2)
        b = add_noise(a, 1, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.DataFrame)
        assert (8760,2) == b.shape

    def test_ndarray_add_noise_uniform(self):
        a = np.random.rand(8760)
        b = add_noise(a, 2, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.Series)

    def test_2d_ndarray_add_noise_uniform(self):
        a = np.random.rand(8760, 2)
        b = add_noise(a, 2, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.DataFrame)
        assert (8760,2) == b.shape

    def test_add_noise_not_annual(self):
        a = np.random.rand(15)
        b = add_noise(a, 3, 0.05)
        assert isinstance(b, pd.Series)


class Test_gen_monthly_daily():

    def test_gen_monthly_daily(self):
        Weight = .55  # Ratio between working and non-working day load (e.g. 70% - 30% )
        ML = 1000 * np.ones(12)  # monthly load
        DWL = np.random.rand(24) * 10  # daily load working
        DWL = DWL / DWL.sum()  # normalized
        DNWL = np.random.rand(24) * 5  # daily load non working
        DNWL = DNWL / DNWL.sum()  # daily load non working
        year = 2014
        Load1 = gen_load_from_daily_monthly(ML, DWL, DNWL, Weight, year)
        assert len(Load1) == 8760
        assert np.isclose(Load1.sum(), np.sum(ML))

class Test_gen_dummy_load():
    def test_gen_dummy(self):
        a = gen_daily_stoch_el(1500)
        assert isinstance(a, np.ndarray)
        assert len(a) == 24

class Test_gen_load_sinus():
    def test_gen_sinus(self):
        Load1 = gen_load_sinus(1,2,3,4,5,6)
        assert len(Load1) == 8760

class Test_disag():
    def test_disag_daily_to_hourly(self):
        x = np.arange(0, 365)
        y = (np.cos(2 * np.pi / 364 * x) * 50 + 100)

        y = make_timeseries(y, freq='D')

        disag_profile = np.random.rand(24)
        y_disag = disag_upsample(y, disag_profile)
        assert np.isclose(np.sum(y_disag), np.sum(y)) # <= 0.001 #FIXME: np test equality
        assert len(y_disag) == 8760

    def test_disag_hourly_to_minutes(self):
        x = np.arange(0, 8760)
        y = (np.cos(2 * np.pi / 8759 * x) * 50 + 100)
        y = make_timeseries(y, freq='h')

        disag_profile = np.random.rand(60)
        y_disag = disag_upsample(y, disag_profile, to_offset='min')
        assert np.isclose(np.sum(y_disag), np.sum(y) ) # <= 0.001 #FIXME: np test equality
        assert len(y_disag) == 8760*60


class Test_demand_side_management():

    def test_load_shifting_small(self):
        a = np.random.rand(8760) * 100
        a = clean_convert(a, force_timed_index=True, always_df=False)
        b = gen_demand_response(a,.1,.2)
        assert np.isclose(np.sum(a), np.sum(b))
        assert np.max(a) > np.max(b)
    def test_load_shifting_big(self):
        a = np.random.rand(8760) * 100
        a = clean_convert(a, force_timed_index=True, always_df=False)
        b = gen_demand_response(a,.15,.5)
        assert np.isclose(np.sum(a), np.sum(b))
        assert np.max(a) > np.max(b)

def test_countweekend_days_per_month():
    a = make_timeseries(year=2015, length=8760, freq='h')
    b = countweekend_days_per_month(a.resample('d').mean())
    assert len(b) == 12
    assert sum(b) == 104 #weekend days in 2015

def test_gen_analytical_LDC():
    #Generate a simple LDC with Peak of 1
    U = (1, 0.5, 0.2, 8760)
    LDC = gen_analytical_LDC(U)

    assert max(LDC[0]) == 1.0
    assert min(LDC[0]) == 0.0
    assert np.isclose(np.mean(LDC[0]), 0.5)


def test_gen_load_from_LDC():
    # Only operate 90% of the time.
    duration_fraction = 0.9
    LDC = gen_analytical_LDC((1, 0.5, 0.2, 8760 * duration_fraction))
    b = gen_load_from_LDC(LDC)
    assert b.max() <= 1.0

    # According to the defined formula anything below should be zero
    val_perc = np.percentile(b, (1 - duration_fraction - 0.01) * 100)
    assert np.isclose(val_perc, 0.0)


def test_gen_corr_arrays():
    Na = 2
    length = 1000
    r = 0.85
    M = np.array([[1, r],
                  [r, 1]])
    A = gen_corr_arrays(Na, length, M)
    new_r = np.corrcoef(A)[0][1]
    assert A.shape == (Na, length)
    #allow some tolerance of convergence..
    assert np.abs(new_r - r) <= 0.03