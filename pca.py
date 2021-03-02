from typing import Tuple, Any

import pandas as pd
import numpy as np


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


def _eigenvalue(df: pd.DataFrame) -> Tuple[Any, Any]:
    w, v = np.linalg.eigh(df)
    print('<<<<_Eigenvalue>>>>>')
    print('input: ')
    print(df)
    print('eigenvalues: ')
    print(w)
    print('eigenvectors: ')
    print(v)

    return w, v


def get_eigenvalue(df: pd.DataFrame) -> Tuple[Any, Any]:
    return _eigenvalue(create_cov_matrix(df))


def pca(df: pd.DataFrame) -> pd.DataFrame:
    w, v = get_eigenvalue(df)
    res = df @ v
    print('<<<<PCA>>>>')
    print(res)
    return res
