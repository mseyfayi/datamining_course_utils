import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from functions import cov_series, create_cov_matrix, _eigenvalue, get_eigenvalue, get_pca, sub_series, \
    correlation_series, correlation_frame, gini_series, weighted_average_of_impurity, get_possibility_series, \
    entropy_series


class TestSubSeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = sr
        assert_series_equal(exp, sub_series(sr), check_dtype=False)

    def test1(self):
        sr = pd.Series([3, 4, 1, 2, 0])  # mean = 2
        exp = pd.Series([1, 2, -1, 0, -2])
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


class TestCorrelationSeries(unittest.TestCase):
    def test_empty(self):
        sr2 = sr1 = pd.Series(dtype='float64')
        exp = 0
        self.assertEqual(exp, correlation_series(sr1, sr2))

    def test_raise_length_error(self):
        x = pd.Series([3, 4, 2, 0])
        y = pd.Series([1, 3, 0, 4, 2])
        try:
            correlation_series(x, y)
            self.fail()
        except ValueError:
            self.assertEqual(True, True)

    def test1(self):
        x = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
        y = pd.Series([1, 3, 0, 4, 2])  # -1, 1, -2, 2, 0
        # xy = 3
        # xx = 10
        # yy = 10
        exp = 3 / 10
        self.assertEqual(exp, correlation_series(x, y))

    def test2(self):
        x = pd.Series([13, 14, 35, 32, 4, 63, 234, 1])
        y = pd.Series([1, 3, 0, 4, 2, 2, 42, 2])

        exp = 0.961177121
        np.testing.assert_almost_equal(exp, correlation_series(x, y))


class TestCorrelationFrame(unittest.TestCase):
    def test_empty(self):
        df = pd.DataFrame()
        exp = df
        assert_frame_equal(exp, correlation_frame(df))

    def test1(self):
        x = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
        y = pd.Series([1, 3, 0, 4, 2])  # -1, 1, -2, 2, 0
        df = pd.DataFrame({'x': x, 'y': y})
        exp = pd.DataFrame([[1, 0.3], [0.3, 1]], columns=['x', 'y'], index=['x', 'y'])
        assert_frame_equal(exp, correlation_frame(df))

    def test2(self):
        x = pd.Series([13, 14, 35, 32, 4, 63, 234, 1])
        y = pd.Series([1, 3, 0, 4, 2, 2, 42, 2])
        df = pd.DataFrame({'x': x, 'y': y})

        e = 0.961177121410483
        exp = pd.DataFrame([[1, e], [e, 1]], columns=['x', 'y'], index=['x', 'y'])
        assert_frame_equal(exp, correlation_frame(df))


class TestGetPossibilitySeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = sr
        assert_series_equal(exp, get_possibility_series(sr))

    def test1(self):
        sr = pd.Series([10, 0])
        exp = pd.Series([1, 0])
        assert_series_equal(exp, get_possibility_series(sr), check_dtype=False)

    def test2(self):
        sr = pd.Series([5, 5])
        exp = pd.Series([1 / 2, 1 / 2])
        assert_series_equal(exp, get_possibility_series(sr), check_dtype=False)

    def test3(self):
        sr = pd.Series([6, 4, 2])
        exp = pd.Series([1 / 2, 1 / 3, 1 / 6])
        assert_series_equal(exp, get_possibility_series(sr), check_dtype=False)


class TestGiniSeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = 1
        self.assertEqual(exp, gini_series(sr))

    def test1(self):
        sr = pd.Series([0, 0, 10])
        exp = 0
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test2(self):
        sr = pd.Series([3, 3, 3])
        exp = 2 / 3
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test3(self):
        sr = pd.Series([5, 4, 3])
        exp = 94 / 144
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test4(self):
        sr = pd.Series([10, 0])
        exp = 0
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test5(self):
        sr = pd.Series([1, 15])
        exp = 15 / 128
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test6(self):
        sr = pd.Series([5, 4])
        exp = 40 / 81
        np.testing.assert_almost_equal(exp, gini_series(sr))

    def test7(self):
        sr = pd.Series([5, 5])
        exp = 1 / 2
        np.testing.assert_almost_equal(exp, gini_series(sr))


class TestGiniFrame(unittest.TestCase):
    def test_empty(self):
        df = pd.DataFrame()
        exp = 0
        self.assertEqual(exp, weighted_average_of_impurity(df, gini_series))

    def test1(self):
        df = pd.DataFrame([
            [5, 4, 3],
            [0, 0, 10],
            [3, 3, 3]
        ])
        exp = 83 / 186
        np.testing.assert_almost_equal(exp, weighted_average_of_impurity(df, gini_series))

    def test2(self):
        df = pd.DataFrame([
            [10, 0],
            [1, 15],
            [5, 4],
            [5, 5]
        ])
        exp = 163 / 648
        np.testing.assert_almost_equal(exp, weighted_average_of_impurity(df, gini_series))


class TestEntropySeries(unittest.TestCase):
    def test_empty(self):
        sr = pd.Series(dtype='float64')
        exp = 0
        self.assertEqual(exp, entropy_series(sr))

    def test1(self):
        sr = pd.Series([0, 0, 10])
        exp = 0
        np.testing.assert_almost_equal(exp, entropy_series(sr))

    def test2(self):
        sr = pd.Series([3, 3, 3])
        exp = 1
        np.testing.assert_almost_equal(exp, entropy_series(sr))

    def test3(self):
        sr = pd.Series([10, 0])
        exp = 0
        np.testing.assert_almost_equal(exp, entropy_series(sr))

    def test4(self):
        sr = pd.Series([5, 9])
        exp = 0.9402859
        np.testing.assert_almost_equal(exp, entropy_series(sr))

    def test5(self):
        sr = pd.Series([5, 5])
        exp = 1
        np.testing.assert_almost_equal(exp, entropy_series(sr))


if __name__ == '__main__':
    unittest.main()
