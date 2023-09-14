import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones, squeeze
from numpy.linalg import inv, multi_dot
from scipy.optimize import minimize

start = time.time()
horizon = 10

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
    phi = np.zeros((assetNum,horizon))
    for t in range(horizon):
        def f(phi, t):
            a = -gamma * square(W[t]) * multi_dot([transpose(phi), covMat, phi])
            b = W[t] * dot(transpose(phi), miu_dis - squeeze(ones((assetNum, 1)) * r))
            c = W[t] * (1 + r)
            return -(a + b + c)

        phi_temp = np.ones((assetNum, 1))
        res = minimize(f, phi_temp, args=(t), method='SLSQP', constraints={'type': 'ineq', 'fun': lambda x: x},
                       options={'disp': False})  # With constraint of no short selling
        phi[:, t] = res.x
        # print(res.x)
        W_new = multiply(W[t], 1 + dot(1 - transpose(phi[:, t]), ones((assetNum, 1)) * r) + dot(transpose(phi[:, t]),
                                                                                                simuReturnList[:, t]))
        W.append(W_new)

    W_T.append(W[-1])


EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))