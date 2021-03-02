from typing import Tuple, Any

import pandas as pd
import numpy as np


def sub_series(sr: pd.Series) -> pd.Series:
    return sr - sr.mean()


def cov_series(sr1: pd.Series, sr2: pd.Series) -> int:
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
