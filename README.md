# Data-mining Course Utils
A personal project to implement Data-mining algorithms in python
## Functions
### PCA
Functions implemented in pca.py file & tests implemented in test/testpca.py
#### sub_series
Gets a pandas series, calculates average (mean) and returns result subtraction of the series and the average
for example:\
sr = [3, 4, 1, 2, 0]\
sr.mean = (3+4+1+2+0)/sr.len = 10/5 = 2\
result = [3-2, 4-2, 1-2, 2-2, 0-2] = [1, 2, -1, 0, -2]