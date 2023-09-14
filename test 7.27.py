import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones, squeeze
from numpy.linalg import inv, multi_dot
from scipy.optimize import minimize
import pandas as pd
import pyEX as p

df = pd.read_excel('Asset Data.xlsx')
df = df.set_index('date')
priceMat = df.to_numpy()
priceMat = np.flip(priceMat)
dfnew = pd.DataFrame(priceMat)
retMat = dfnew.pct_change().dropna().to_numpy()



miu_dis = np.mean(retMat, axis=0)
covMat = np.cov(retMat.T)

print(miu_dis)
print (covMat)


# print (covMat)
# print (miu_dis)


