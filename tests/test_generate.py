import numpy as np
import pandas as pd
from enload import (add_noise, gen_load_from_daily_monthly, gen_load_sinus, gen_demand_response,
                    disag_upsample, make_timeseries, clean_convert, countweekend_days_per_month)

class Test_noise():

    def test_ndarray_add_noise_gauss(self):
        a = np.random.rand(8760)
        b = add_noise(a, 3, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.Series)

    def test_2d_ndarray_add_noise_gauss(self):
        a = np.random.rand(8760, 2)
        b = add_noise(a, 3, 0.05)  # Gauss Markov noise
        assert isinstance(b, pd.DataFrame)

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

class Test_gen_load_sinus():
    def test_gen_sinus(self):
        Load1 = gen_load_sinus(1,2,3,4,5,6)
        assert len(Load1) == 8760

class Test_disag():
    def test_disag_daily_to_hourly(self):
        x = np.arange(0, 365)
        y = (np.cos(2 * np.pi / 364 * x) * 50 + 100)

        y = make_timeseries(y, freq='d')

        disag_profile = np.random.rand(24)
        y_disag = disag_upsample(y, disag_profile)
        assert np.isclose(np.sum(y_disag), np.sum(y)) # <= 0.001 #FIXME: np test equality
        assert len(y_disag) == 8760

    def test_disag_hourly_to_minutes(self):
        x = np.arange(0, 8760)
        y = (np.cos(2 * np.pi / 8759 * x) * 50 + 100)
        y = make_timeseries(y, freq='h')

        disag_profile = np.random.rand(60)
        y_disag = disag_upsample(y, disag_profile, to_offset='t')
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
