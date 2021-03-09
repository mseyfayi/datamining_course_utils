from typing import Tuple, List, Callable

import numpy as np
import pandas as pd


def sub_series(sr: pd.Series) -> pd.Series:
    """
    Gets a pandas series, calculates average (mean) and returns result subtraction of the series and the average
    for example:
    sr = 3, 4, 1, 2, 0
    sr.mean = (3+4+1+2+0)/sr.len = 10/5 = 2
    result = 3-2, 4-2, 1-2, 2-2, 0-2 = 1, 2, -1, 0, -2
    :param sr: Series to calculate (pandas.Series)
    :return: sr - sr.mean() (pandas.Series)
    """
    return sr - sr.mean()


def cov_series(sr1: pd.Series, sr2: pd.Series) -> int:
    """
    Gets to pandas series and returns covariance of them
    Covariance => Cov(a,b) = Sigma(i:0 -> n)(ai-~a)(bi-~b) / n
    Length of sr1 & sr2 must be equal
    for example:
    sr1 = pd.Series([3, 4, 1, 2, 0])  # 1, 2, -1, 0, -2
    sr2 = pd.Series([1, 3, 0, 4, 2])  # -1, 1, -2, 2, 0
    result = ((1*-1) + (2*1) + (-1*-2) + (0*2) + (-2*0)) / sr1.len
        = (-1 + 2 + 2 + 0 + 0) / 5 = 3 / 5 = 0.6
    :param sr1: First series (pandas.Series)
    :param sr2: Second series (pandas.Series)
    :return: Covariance of sr1 & sr2 (int)
    """
    if len(sr1) != len(sr2):
        raise ValueError('Length of sr1 & sr2 must be equal!')
    if len(sr1) == 0:
        return 0
    print('<<<<Sub Covariance>>>>')
    print('sr1: ', list(sr1))
    print('sr2: ', list(sr2))
    sr1 = sub_series(sr1)
    sr2 = sub_series(sr2)
    print("sr1': ", list(sr1))
    print("sr2': ", list(sr2))
    pro = sr1 * sr2
    print("pro: ", list(pro))
    res = sum(pro) / len(pro)
    print('res: ', res)
    print('-----------------')
    return res


def create_cov_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a Covariance Matrix from a matrix (pandas.Dataframe)

        0       1       ... n
    0   Cov00   Cov01       Cov0n
    1   Cov10   Cov11       Cob1n
    .                       .
    .                       .
    .                       .
    n   Covn0   Covn1   ... Covnn

    for example:
    input:
       1  2
    0  3  1
    1  4  3
    2  1  0
    3  2  4
    4  0  2
    output:
         1     2
    1  2.0   0.6
    2  0.6   2.0

    :param df: Input matrix (pandas.Dataframe)
    :return: Covariance matrix (pandas.Dataframe)
    """
    columns = df.columns
    data = {}
    for c1 in columns:
        data[c1] = []
        for c2 in columns:
            data[c1].append(cov_series(df[c1], df[c2]))
    frame = pd.DataFrame(data, columns=columns, index=columns)
    print('<<<Covariance Matrix>>>')
    print(frame)
    return frame


def _eigenvalue(df: pd.DataFrame) -> Tuple[List[int], List[List[int]]]:
    """
    Gets a Matrix (pandas.Dataframe) and returns descending sorted 'eigenvalue' and 'eigenvectors'
    This function using 'numpy.linalg.eigh'
    :param df: Input Matrix (pandas.Dataframe)
    :return: (eigenvalue, eigenvectors) as a Tuple
    """
    w, v = np.linalg.eigh(df)
    w = w[::-1]
    v = v[:, ::-1]
    print('<<<<_Eigenvalue>>>>>')
    print('input: ')
    print(df)
    print('eigenvalues: ')
    print(w)
    print('eigenvectors: ')
    print(v)

    return w, v


def get_eigenvalue(df: pd.DataFrame) -> Tuple[List[int], List[List[int]]]:
    """
    Gets a matrix (pandas.Dataframe), transforms it to its Covariance Matrix and returns 'eigenvalue' and 'eigenvectors'
    :param df: Input Matrix (pandas.Dataframe)
    :return: (eigenvalue, eigenvectors) as a Tuple
    """
    return _eigenvalue(create_cov_matrix(df))


def get_pca(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Gets a Matrix (pandas.Dataframe) and returns result of PCA of it
    :param df: The input matrix (pandas.Dataframe)
    :param threshold: Threshold of eigenvalues to drop lowers (float)
    :return: PCA of input matrix (pandas.Dataframe)
    """
    w, v = get_eigenvalue(df)
    threshold_index = next((i for i, x in enumerate(w) if x < threshold), None)
    print(v)

    if threshold_index:
        v = np.delete(v, np.s_[threshold_index:], axis=1)
    print('<<<<PCA>>>>')

    print('input')
    print(df)
    print('Transform Matrix:')
    print(v)
    res = df @ v
    print('result: ')
    print(res)
    return res


def correlation_series(x: pd.Series, y: pd.Series) -> float:
    """
    Gets two pandas.series and calculates their Pearson Correlation
    Pearson Correlation Formula:
        Sigma(x-~x)(y-~y)/sqrt(Sigma(x-~x)^2.Sigma(y-~y)^2)
    :param x: First Series
    :param y: Second Series
    :return: Pearson Correlation between x & y
    """
    if len(x) != len(y):
        raise ValueError('Length of x & y is not equal')
    if len(x) == 0:
        return 0
    print('<<<<Correlation Series>>>>')
    x = sub_series(x)
    y = sub_series(y)
    xx = sum(x ** 2)
    yy = sum(y ** 2)
    xy = sum(x * y)
    print('xx', xx)
    print('yy', yy)
    print('xy', xy)
    result = xy / np.sqrt(xx * yy)
    print('result: ', result)
    return result

###########################################

def correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets a matrix (pandas.Dataframe) and returns correlation matrix of its columns
    :param df: Input matrix (pandas.Dataframe)
    :return: correlation matrix (pandas.Dataframe)
    """
    columns = df.columns
    data = {}
    for c1 in columns:
        data[c1] = []
        for c2 in columns:
            data[c1].append(correlation_series(df[c1], df[c2]))
    frame = pd.DataFrame(data, index=columns)
    print('<<<<Correlation Frame>>>>')
    print('input: ')
    print(df)
    print('output:')
    print(frame)
    return frame

###########################################

def weighted_average_of_impurity(df: pd.DataFrame, impurity_func: Callable[[pd.Series], float]) -> float:
    col_sum = df.sum(axis=1)
    col_gini = df.apply(impurity_func, axis=1)
    res = sum(col_gini * col_sum / sum(col_sum))
    print('<<<<Frame Gini>>>>')
    print('Sum of Columns')
    print(col_sum)
    print('Gini of Columns')
    print(col_gini)
    print('Result')
    print(res)

    return res


def get_possibility_series(sr: pd.Series) -> pd.Series:
    return sr / sum(sr)


def gini_series(sr: pd.Series) -> float:
    p = get_possibility_series(sr)
    p2 = p * p
    res = 1 - sum(p2)
    print('<<<<Gini>>>>')
    print('input:')
    print(sr)
    print('Possibility:')
    print(p)
    print('Result:')
    print(res)
    return res


def entropy_series(sr: pd.Series) -> float:
    p = get_possibility_series(sr)
    n = len(sr)
    log_p = (np.log(p) / np.log(n)).replace(-np.inf, 0)
    p_log_p = p * log_p
    res = -sum(p_log_p)
    print('<<<<Entropy>>>>')
    print('input:')
    print(sr)
    print('Possibility:')
    print(p)
    print('Result:')
    print(res)
    return res


def classification_error_series(sr: pd.Series) -> float:
    p = get_possibility_series(sr)
    return 1 - p.max()
