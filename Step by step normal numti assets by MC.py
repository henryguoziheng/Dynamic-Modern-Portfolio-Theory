import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones
from numpy.linalg import inv, multi_dot

start = time.time()
horizon = 10

assetNum = 2 # assume returns are independent
miu_dis = np.array([0.034, -0.015]) # length is the same as asset numbers
sigma = np.array([0.09308061, 0.105]) # std

w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate


W_T = []
pathNum = 100000
for path in range(pathNum):
    simuReturnList = np.random.normal(miu_dis[0:horizon], sigma[0:horizon], horizon)
    #print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon):  # t = 0,1,2,3,...,9
        phi_new = (E(miu_dis[t], sigma[t]) - r)/(2*gamma*W[t]*Var(miu_dis[t], sigma[t]))
        phi_new = ()/(2*gamma*W[t])
        phi.append(phi_new)
        W_new = multiply(W[t], 1 + (1-phi[t])*r + phi[t]*simuReturnList[t])
        #W_new = W[t] * (1 + (1 - phi[t]) * r + phi[t] * simuReturnList[t])
        #print (W_new)
        W.append(W_new)

    W_T.append(W[-1])


EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))