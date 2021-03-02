import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pca import sub_series, cov_series, create_cov_matrix


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

    def test4(self):
        sr1 = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
        sr2 = pd.Series([1, 3, 0, 4, 2])  # -1, 1, -2, 2, 0
        exp = 3 / 5
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test5(self):
        sr1 = pd.Series([3, 4, 1, 2, 0])  # 1,  2, -1,  0, -2
        sr2 = pd.Series([5, -4, 10, -11, 5])  # 4, -5,  9, -12, 4
        exp = -23 / 5
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test6(self):
        sr1 = pd.Series([1, 3, 0, 4, 2])  # -1, 1, -2, 2, 0
        sr2 = pd.Series([5, -4, 10, -11, 5])  # 4, -5,  9, -12, 4
        exp = -51 / 5
        self.assertEqual(exp, cov_series(sr1, sr2))

    def test7(self):
        sr1 = pd.Series([5, -4, 10, -11, 5])  # 4, -5,  9, -12, 4
        sr2 = pd.Series([5, -4, 10, -11, 5])  # 4, -5,  9, -12, 4
        exp = 282 / 5
        self.assertEqual(exp, cov_series(sr1, sr2))


class TestCreateCovMatrix(unittest.TestCase):
    def test_empty(self):
        df = pd.DataFrame()
        exp = pd.DataFrame()
        assert_frame_equal(exp, create_cov_matrix(df), check_dtype=False)

    def test1(self):
        df = pd.DataFrame({'1': [3, 4, 1, 2, 0], '2': [3, 4, 1, 2, 0]})
        exp = pd.DataFrame({'1': [2, 2], '2': [2, 2]}, index=['1', '2'])
        assert_frame_equal(exp, create_cov_matrix(df), check_dtype=False, check_index_type=False)

    def test2(self):
        df = pd.DataFrame({'1': [3, 4, 1, 2, 0], '2': [1, 3, 0, 4, 2], '3': [5, -4, 10, -11, 5]})
        exp = pd.DataFrame({'1': [2, 3 / 5, -23 / 5], '2': [3 / 5, 2, -51 / 5], '3': [-23 / 5, -51 / 5, 282 / 5]},
                           index=['1', '2', '3'])
        assert_frame_equal(exp, create_cov_matrix(df), check_dtype=False, check_index_type=False)


if __name__ == '__main__':
    unittest.main()
