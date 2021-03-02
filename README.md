# Data-mining Course Utils

A personal project to implement Data-mining algorithms in python

## Functions

### PCA

> Functions implemented in pca.py file & tests implemented in test/testpca.py

#### sub_series

> - Gets a pandas series, calculates average (mean) and returns result subtraction of the series and the average
> - for example:
>>    sr = [3, 4, 1, 2, 0]\
>>    sr.mean = (3+4+1+2+0)/sr.len = 10/5 = 2\
>>    result = [3-2, 4-2, 1-2, 2-2, 0-2] = [1, 2, -1, 0, -2]

#### cov_series

> - Gets to pandas series and returns covariance of them
> - Covariance =>![Covariance formula](https://cdn.corporatefinanceinstitute.com/assets/covariance1.png) 
> - Length of sr1 & sr2 must be equal
> - for example:
>>    sr1 = pd.Series([3, 4, 1, 2, 0])\
>>    sr1 - sr1.mean = 1, 2, -1, 0, -2\
>>    sr2 = pd.Series([1, 3, 0, 4, 2])\
>>    sr2 - sr2.mean = -1, 1, -2, 2, 0\
>>    result = ((1*-1) + (2*1) + (-1*-2) + (0*2) + (-2*0)) / sr1.len  
>>    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (-1 + 2 + 2 + 0 + 0) / 5 = 3 / 5 = 0.6