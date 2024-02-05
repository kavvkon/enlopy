import numpy as np
import pandas as pd

from enlopy.utils import make_timeseries , clean_convert, _freq_to_sec


class Test_make_timeseries():

    def test_ndarray_h(self):
        a = np.random.rand(8760)
        b = make_timeseries(a)
        assert isinstance(b, pd.Series)
        assert np.isclose(sum(a), sum(b))

    def test_ndarray_15m(self):
        a = np.random.rand(35040)
        b = make_timeseries(a)
        assert isinstance(b, pd.Series)
        assert np.isclose(sum(a), sum(b))

    def test_ndarray_x(self):
        a = np.random.rand(8730)
        b = make_timeseries(a, freq='h')
        assert isinstance(b, pd.Series)
        assert np.isclose(sum(a), sum(b))

    def test_pdseries(self):
        a = pd.Series(np.random.rand(8760))
        b = make_timeseries(a)
        assert isinstance(b, pd.Series)
        assert np.isclose(sum(a), sum(b))

    def test_pddataframe(self):
        a = pd.DataFrame(np.random.rand(8760))
        b = make_timeseries(a)
        assert isinstance(b, pd.DataFrame)

    def test_pddataframe_2d(self):
        a = pd.DataFrame(np.random.rand(35040,3))
        b = make_timeseries(a)
        assert isinstance(b, pd.DataFrame)

    def test_2d_ndarray(self):
        a = np.random.rand(8760,5)
        b = make_timeseries(a)
        assert isinstance(b, pd.DataFrame)

    def test_empty_frame(self):
        a = np.array([])
        b = make_timeseries(a, freq='h')
        assert isinstance(b, pd.Series) and len(b)==0

    def test_empty_frame_to_indexed_empty(self):
        b = make_timeseries(freq='h', length=8760)
        assert isinstance(b, pd.Series) and len(b) == 8760

    def test_multiannual_timeseries(self):
        a = np.random.rand(8760*2)
        b = make_timeseries(a, freq='h')
        assert isinstance(b, pd.Series)
        assert np.isclose(sum(a), sum(b))


class Test_clean_convert():

    def test_ndarray_to_series_indexed(self):
        a = np.random.rand(8760)
        b = clean_convert(a, force_timed_index=True, always_df=False)
        assert isinstance(b, pd.Series) and isinstance(b.index, pd.DatetimeIndex)

    def test_ndarray_to_df_indexed(self):
        a = np.random.rand(8760)
        b = clean_convert(a, force_timed_index=True, always_df=True)
        assert isinstance(b, pd.DataFrame) and isinstance(b.index, pd.DatetimeIndex)

    def test_1d_series_to_frame(self):
        a = pd.Series(np.random.rand(8760))
        b = clean_convert(a, force_timed_index=True, always_df=True)
        assert isinstance(b, pd.DataFrame) and isinstance(b.index, pd.DatetimeIndex)

    def test_2d_ndarray_to_df_indexed(self):
        a = np.random.rand(8760, 2)
        b = clean_convert(a, force_timed_index=True, always_df=True)
        assert isinstance(b, pd.DataFrame) and isinstance(b.index, pd.DatetimeIndex)

    def test_list_to_series(self):
        a = list(np.random.rand(8760))
        b = clean_convert(a, force_timed_index=True, always_df=False)
        assert isinstance(b, pd.Series)


def test_freq_to_sec():
    a = _freq_to_sec('d')
    assert a == 60 * 60 * 24