import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp
from numpy.linalg import inv

start = time.time()
horizon = 10
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

#print (E(miu, sigma))

def b_temp(t): # if asset number > 1, then miu[t] change to miu[:, t]
    return square(E(miu_dis[t], sigma[t]) - r) / (Var(miu_dis[t], sigma[t]) + square(E(miu_dis[t], sigma[t]) - r))


A = np.ones((3, horizon))
for t in range(0, horizon):
    A[1, t] = (1+r)*(1-b_temp(t))
    A[2, t] = power(1+r, 2)*(1-b_temp(t))

B = np.ones((3, horizon))
for t in range(0, horizon):
    if t < horizon:
        B[1, t] = b_temp(t) * prod(A[1, t+1:])/(2*prod(A[2, t+1:]))
        B[2, t] = b_temp(t) * power(prod(A[1, t+1:])/(2*prod(A[2, t+1:])), 2)
    if t == horizon:
        B[1, t] = b_temp(t) * (1/2)
        B[2, t] = b_temp(t) * (1/4)

miu = prod(A[1, :])
temp2 = np.append(cumprod(A[1, 1:][::-1])[::-1], 1)
nu = dot(temp2, B[1, :])
tau = prod(A[2, :])
a = nu/2 - power(nu ,2)
b = (miu * nu) / a
c = tau - power(miu, 2) - a*power(b, 2)


W_T = []
pathNum = 100000
for path in range(pathNum):
    simuReturnList = np.random.normal(miu_dis[0:horizon], sigma[0:horizon], horizon)
    #print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon):  # t = 0,1,2,3,...,9
        temp = (E(miu_dis[t], sigma[t]) - r) / (Var(miu_dis[t], sigma[t]) + square(E(miu_dis[t], sigma[t]) - r))
        if t < horizon-1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t+1:horizon])) * temp) / W[t] # find control at time t
            phi.append(phi_new) # append control at time t
            W_new = multiply(W[t], 1 + (1-phi[t])*r + phi[t]*simuReturnList[t]) # find wealth at t+1 according to control at t
            W.append(W_new)
        if t == horizon-1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * temp) / W[t]
            phi.append(phi_new)
            W_new = multiply(W[t], 1 + (1-phi[t])*r + phi[t]*simuReturnList[t])
            W.append(W_new)
    #print (W)
    W_T.append(W[-1])

EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))


