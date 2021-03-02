import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pca import sub_series, cov_series


class TestSubSeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = sr
        assert_series_equal(exp, sub_series(sr), check_dtype=False)

    def test1(self):
        sr = pd.Series([3, 4, 1, 2, 0])  # mean = 2
        exp = pd.Series([1, 2, -1, 0, -2])  # mean = 20
        assert_series_equal(exp, sub_series(sr), check_dtype=False)


class TestCvoSeries(unittest.TestCase):
    def test_empty(self):
        sr1 = sr2 = pd.Series(dtype='float64')
        exp = 0
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test_raise_def_length(self):
        sr1 = pd.Series([1, 2, 3])
        sr2 = pd.Series([1, 2])

        try:
            cov_series(sr1, sr2)
            self.fail()
        except ValueError:
            self.assertEqual(True, True)

    def test1(self):
        sr1 = pd.Series([2, 4])  # -1, 1
        sr2 = pd.Series([4, 2])  # 1, -1
        exp = -1
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test2(self):
        sr1 = pd.Series([0])  # -1, 1
        sr2 = pd.Series([0])  # 1, -1
        exp = 0
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test3(self):
        sr1 = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
        sr2 = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
        exp = 2
        self.assertEqual(exp, cov_series(sr1, sr2))


if __name__ == '__main__':
    unittest.main()
