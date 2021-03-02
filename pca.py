import pandas as pd


def sub_series(sr: pd.Series) -> pd.Series:
    return sr - sr.mean()


def cov_series(sr1: pd.Series, sr2: pd.Series) -> int:
    if len(sr1) != len(sr2):
        raise ValueError('Length of sr1 & sr2 must be equal!')
    if len(sr1) == 0:
        return 0
    sr1 = sub_series(sr1)
    sr2 = sub_series(sr2)
    pro = sr1 * sr2
    return sum(pro) / len(pro)
