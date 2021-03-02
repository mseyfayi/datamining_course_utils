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
