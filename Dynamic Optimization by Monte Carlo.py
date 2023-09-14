import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide

start = time.time()
horizon = 10
p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate


def b_temp(t):
    return np.square(np.dot(np.reshape(r_list[:,t] - r * np.ones(2), (1,2)), p_list[:,t])) / np.dot(np.square(np.reshape(r_list[:,t] - r * np.ones(2), (1,2))), p_list[:,t])

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

# Start Monte Carlo
W_T = [] # records wealth at T for different path
pathNum = 100000 # Total number of pathes for Monte Carlo
for path in range(pathNum):
    # Select a stock return path according to the probability at each time step
    slctVec = np.random.binomial(1, p_list[1, 0:horizon])
    r_list_new = np.transpose(r_list[:, 0:horizon])
    simuReturnList = r_list_new[np.arange(len(r_list_new)), slctVec]

    W = [10] # initial wealth
    phi = []
    for t in range(0, horizon): # t = 0,1,2,3,...,9
        temp = np.dot(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2)), p_list[:, t]) / np.dot(np.square(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2))), p_list[:, t])
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
    #print(W)
    W_T.append(W[-1])

EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))
