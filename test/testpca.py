import unittest

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from pca import cov_series, create_cov_matrix, _eigenvalue, get_eigenvalue, get_pca


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

    def test3(self):
        df = pd.DataFrame({'1': [3, 4, 1, 2, 0], '2': [1, 3, 0, 4, 2]})
        print(df)
        exp = pd.DataFrame({'1': [2, 3 / 5], '2': [3 / 5, 2]},
                           index=['1', '2'])
        assert_frame_equal(exp, create_cov_matrix(df), check_dtype=False, check_index_type=False)


class TestTempEigenvalue(unittest.TestCase):
    def test1(self):
        df = pd.DataFrame({'1': [2, 0], '2': [0, 1]},
                          index=['1', '2'])
        w, v = _eigenvalue(df)
        expw = np.array([2, 1])
        expv = np.array([[1, 0], [0, 1]])
        assert_array_equal(expw, w)
        assert_array_equal(expv, v)

    def test2(self):
        df = pd.DataFrame({'1': [2, 0], '2': [0, 2]},
                          index=['1', '2'])
        w, v = _eigenvalue(df)
        expw = np.array([2, 2])
        expv = np.array([[0, 1], [1, 0]])
        assert_array_equal(expw, w)
        assert_array_equal(expv, v)

    def test3(self):
        df = pd.DataFrame({'1': [2, 0, 0], '2': [0, 2, 0], '3': [0, 0, 2]},
                          index=['1', '2', '3'])
        w, v = _eigenvalue(df)
        expw = np.array([2, 2, 2])
        expv = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert_array_equal(expw, w)
        assert_array_equal(expv, v)

    def test4(self):
        df = pd.DataFrame({'1': [3, 0, 0], '2': [0, 4, 0], '3': [0, 0, 0]},
                          index=['1', '2', '3'])

        w, v = _eigenvalue(df)

        expw = np.array([4, 3, 0])
        expv = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        assert_array_equal(expw, w)
        assert_array_equal(expv, v)


class TestEigenvalue(unittest.TestCase):
    def test1(self):
        df = pd.DataFrame({'1': [3, 4, 1, 2, 0], '2': [3, 4, 1, 2, 0]})
        w, v = get_eigenvalue(df)
        expw = np.array([4, 0])
        sqr2inv = 1 / np.sqrt(2)
        expv = np.array([[sqr2inv, -sqr2inv], [sqr2inv, sqr2inv]])
        assert_array_equal(expw, w)
        assert_array_equal(expv, v)


class TestPCA(unittest.TestCase):
    def test1(self):
        l = [3, 4, 1, 2, 0]
        df = pd.DataFrame({'1': l, '2': l})
        sqr2 = np.sqrt(2)
        exp = pd.DataFrame([[sqr2 * x] for x in l])
        assert_frame_equal(exp, get_pca(df), check_column_type=False)


if __name__ == '__main__':
    unittest.main()
