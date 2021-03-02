import unittest
import pandas as pd

from pandas.testing import assert_series_equal

from utils import sub_series


class TestSubSeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = sr
        assert_series_equal(exp, sub_series(sr), check_dtype=False)

    def test1(self):
        sr = pd.Series([3, 4, 1, 2, 0])  # mean = 2
        exp = pd.Series([1, 2, -1, 0, -2])
        assert_series_equal(exp, sub_series(sr), check_dtype=False)
