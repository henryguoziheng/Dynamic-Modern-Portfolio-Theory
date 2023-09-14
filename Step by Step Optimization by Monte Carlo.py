import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square

start = time.time()
horizon = 10
p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate

def Er(t):
    return dot(r_list[:, t], p_list[:, t])

def ErSqr(t):
    return dot(square(r_list[:, t]), p_list[:, t])

W_T = []
pathNum = 100000
for path in range(pathNum):
    slctVec = np.random.binomial(1, p_list[1, 0:horizon])
    r_list_new = np.transpose(r_list[:, 0:horizon])
    simuReturnList = r_list_new[np.arange(len(r_list_new)), slctVec]

    W = [10]
    phi = []
    for t in range(0, horizon): # t = 0,1,2,3,...,9

        phi_new = (W[t]*(Er(t)-r))/(2*gamma*power(W[t], 2)*(ErSqr(t) - power(Er(t), 2)))
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