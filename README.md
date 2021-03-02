# Data-mining Course Utils

A personal project to implement Data-mining algorithms in python

## Functions

- [sub_series](#sub_series)
- [cov_series](#cov_series)
- [create_cov_matrix](#create_cov_matrix)
- [_eigenvalue](#_eigenvalue)
- [get_eigenvalue](#get_eigenvalue)
- [get_pca](#get_pca)
- Tests: [test.py](test/test.py)

### sub_series

- Gets a pandas series, calculates average (mean) and returns result subtraction of the series and the average
- for example:

> sr = [3, 4, 1, 2, 0]\
> sr.mean = (3+4+1+2+0)/sr.len = 10/5 = 2\
> result = [3-2, 4-2, 1-2, 2-2, 0-2] = [1, 2, -1, 0, -2]

### cov_series

- Gets to pandas series and returns covariance of them
- Covariance =>\
  ![Covariance formula](https://cdn.corporatefinanceinstitute.com/assets/covariance1.png)
- Length of sr1 & sr2 must be equal
- for example:

> sr1 = pd.Series([3, 4, 1, 2, 0])\
> sr1 - sr1.mean = 1, 2, -1, 0, -2\
> sr2 = pd.Series([1, 3, 0, 4, 2])\
> sr2 - sr2.mean = -1, 1, -2, 2, 0\
> result = ((1*-1) + (2*1) + (-1*-2) + (0*2) + (-2*0)) / sr1.len  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (-1 + 2 + 2 + 0 + 0) / 5 = 3 / 5 = 0.6

### create_cov_matrix

Creates a Covariance Matrix from a matrix (pandas.Dataframe)

Covariance Matrix:

|           | 0                | 1                | ...              | n                |
| --------- | :--------------: | :--------------: | :--------------: | :--------------: |
| **1**     | Cov<sub>00</sub> | Cov<sub>01</sub> |                  | Cov<sub>0n</sub> |
|  ...      |                  |                  |                  |                  |
| **n**     | Cov<sub>n0</sub> | Cov<sub>n1</sub> | ...              | Cov<sub>nn</sub> |

**n** is number od columns\
Cov<sub>ij</sub> is Covariance of column<sub>i</sub> & column<sub>j</sub>

for example:\
input matrix:

|           | 1        | 2      | 
| --------- | :------: | :----: | 
| **0**     | 3        | 1      | 
| **1**     | 4        | 3      | 
| **2**     | 1        | 0      | 
| **3**     | 2        | 4      | 
| **4**     | 0        | 2      | 

output:

|           | 1        | 2      | 
| --------- | :------: | :----: | 
| **1**     | 2        | 0.6    | 
| **2**     | 0.6      | 2      | 

### _eigenvalue

- Gets a Matrix (pandas.Dataframe) and returns descending
  sorted [eigenvalue & eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
- This function using [numpy.linalg.eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)

### get_eigenvalue

1. Gets a matrix (pandas.Dataframe)
2. Transforms it to its Covariance Matrix (using create_cov_matrix)
3. Returns 'eigenvalue' and 'eigenvectors' (using _eigenvalue)

### get_pca

Gets a Matrix (pandas.Dataframe) and returns the PCA
