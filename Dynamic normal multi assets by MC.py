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

def b_temp(t):
    temp1 = transpose(miu_dis - r)
    temp2 = covMat + np.outer(temp1, temp1)
    temp3 = (miu_dis - r)
    return multi_dot([temp1, inv(temp2), temp3])

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
pathNum = 10000
for path in range(pathNum):
    simuReturnList = transpose(np.random.multivariate_normal(miu_dis, covMat, horizon))
    #print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon):  # t = 0,1,2,3,...,9
        temp1 = covMat + np.outer(miu_dis - r, miu_dis - r)
        temp2 = (miu_dis - r)
        temp = dot(inv(temp1), temp2)
        if t < horizon - 1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t + 1:horizon])) * temp) / W[t]  # find control at time t
            phi.append(phi_new)  # append control at time t
            W_new = multiply(W[t], 1 + dot(1 - transpose(phi[t]), ones((assetNum, 1)) * r) + dot(transpose(phi[t]), simuReturnList[:, t]))
            W.append(W_new)
        if t == horizon - 1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * temp)/W[t]
            phi.append(phi_new)  # append control at time t
            W_new = multiply(W[t], 1 + dot(1 - transpose(phi[t]), ones((assetNum, 1)) * r) + dot(transpose(phi[t]), simuReturnList[:, t]))
            W.append(W_new)
    #print (W)
    #print (phi)
    W_T.append(W[-1])


EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))