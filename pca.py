import pandas as pd


def sub_series(sr: pd.Series) -> pd.Series:
    return sr - sr.mean()





