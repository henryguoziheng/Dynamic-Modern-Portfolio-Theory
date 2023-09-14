import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones
from numpy.linalg import inv, multi_dot

start = time.time()
horizon = 2

# assume returns are independent
miu_dis = np.array([0.034, -0.015]) # length is the same as asset numbers
covMat = np.array([[0.008649, 0],
                  [0, 0.011025]])  # covariance matrix
assetNum = np.shape(miu_dis)[0]

w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate


W_T = []
pathNum = 10000
for path in range(pathNum):
    simuReturnList = transpose(np.random.multivariate_normal(miu_dis, covMat, horizon))
    #print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon):  # t = 0,1,2,3,...,9
        phi_new = dot(inv(covMat), miu_dis - r)/(2*gamma*W[t])
        phi.append(phi_new)
        W_new = multiply(W[t], 1 + dot(1 - transpose(phi[t]), ones((assetNum, 1)) * r) + dot(transpose(phi[t]), simuReturnList[:, t]))
        W.append(W_new)

    #print (phi)
    #print (W)
    W_T.append(W[-1])

EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))