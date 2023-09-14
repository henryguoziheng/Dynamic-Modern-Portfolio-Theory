import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp

start = time.time()
horizon = 1

# miu_dis = np.array([0.03, -0.05, 0.04, 0.08, -0.07, 0.10, -0.03, -0.05, 0.06, 0.02])
# sigma = np.array([0.25, 0.2, 0.33, 0.55, 0.15, 0.10, 0.16, 0.2, 0.3, 0.1])

miu_dis = np.array([0.034, -0.015, 0.078, -0.0025, -0.0125, -0.061, 0.0105, -0.025, -0.017, 0.0225])
sigma = np.array([0.09308061, 0.105, 0.084, 0.07462406, 0.0475, 0.10539924,0.05472431, 0.055, 0.05040833, 0.07462406])

w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate

def E(miu, sigma):
    return miu

def Var(miu, sigma):
    return square(sigma)


W_T = []
pathNum = 100000
for path in range(pathNum):
    simuReturnList = np.random.normal(miu_dis[0:horizon], sigma[0:horizon], horizon)
    #print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon):  # t = 0,1,2,3,...,9
        phi_new = (E(miu_dis[t], sigma[t]) - r)/(2*gamma*W[t]*Var(miu_dis[t], sigma[t]))
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
