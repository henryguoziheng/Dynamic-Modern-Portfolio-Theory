import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones
from numpy.linalg import inv, multi_dot
import matplotlib.pyplot as plt

assetNum = 2  # assume returns are independent
miu_dis = np.array([0.034, -0.015])  # length is the same as asset numbers
sigma = np.array([0.093, 0.105])  # std

w0 = 10  # W(0) starting wealth
r = 0.005  # risk-free rate
gamma = 0.5  # risk aversion rate. E(X)-gamma*Var(X)

timeList = [1,2,3,4,5]
MeanVar_Dynamic = []
MeanVar_Sbs = []
for horizon in timeList:

    def b_temp(t):
        temp1 = transpose(miu_dis - r)
        temp2 = np.array([[square(sigma[0]) + square(miu_dis[0]-r), (miu_dis[0]-r)*(miu_dis[1]-r)],
                          [(miu_dis[0]-r)*(miu_dis[1]-r), square(sigma[1]) + square(miu_dis[1]-r)]])
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


    W_T_Dynamic = []
    W_T_Sbs = []
    pathNum = 10000
    for path in range(pathNum):
        simuReturnList = transpose(np.random.normal(miu_dis, sigma, [horizon, assetNum]))
        #print (simuReturnList)

        W_Dynamic = [10]
        W_Sbs = [10]
        phi_Dynamic = []
        phi_Sbs = []
        for t in range(0, horizon):  # t = 0,1,2,3,...,9
            temp1 = np.array([[square(sigma[0]) + square(miu_dis[0] - r), (miu_dis[0] - r) * (miu_dis[1] - r)],
                              [(miu_dis[0] - r) * (miu_dis[1] - r), square(sigma[1]) + square(miu_dis[1] - r)]])
            temp2 = (miu_dis - r)
            temp = dot(inv(temp1), temp2)
            if t < horizon - 1:
                phi_new_Dynamic = (-(1 + r) * W_Dynamic[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t + 1:horizon])) * temp) / W_Dynamic[t]  # find control at time t
                phi_Dynamic.append(phi_new_Dynamic)  # append control at time t
                W_new = multiply(W_Dynamic[t], 1 + dot(1 - transpose(phi_Dynamic[t]), ones((assetNum, 1)) * r) + dot(transpose(phi_Dynamic[t]), simuReturnList[:, t]))
                W_Dynamic.append(W_new)
            if t == horizon - 1:
                phi_new_Dynamic = (-(1 + r) * W_Dynamic[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * temp)/W_Dynamic[t]
                phi_Dynamic.append(phi_new_Dynamic)  # append control at time t
                W_new = multiply(W_Dynamic[t], 1 + dot(1 - transpose(phi_Dynamic[t]), ones((assetNum, 1)) * r) + dot(transpose(phi_Dynamic[t]), simuReturnList[:, t]))
                W_Dynamic.append(W_new)

            phi_new_Sbs = [(miu_dis[0] - r) / (2 * gamma * W_Sbs[t] * square(sigma[0])),
                       (miu_dis[1] - r) / (2 * gamma * W_Sbs[t] * square(sigma[1]))]
            phi_Sbs.append(phi_new_Sbs)
            W_new_Sbs = multiply(W_Sbs[t], 1 + dot(1 - transpose(phi_Sbs[t]), ones((assetNum, 1)) * r) + dot(transpose(phi_Sbs[t]), simuReturnList[:, t]))
            W_Sbs.append(W_new_Sbs.flatten())

        #print (W)
        #print (phi)
        W_T_Dynamic.append(W_Dynamic[-1])
        W_T_Sbs.append(W_Sbs[-1])


    EWt_dynamic = np.mean(W_T_Dynamic)
    VarWt_dynamic = np.var(W_T_Dynamic)
    MeanVar_Dynamic.append(EWt_dynamic - gamma*VarWt_dynamic)
    print ('Dynamic E[Wt] - gamma*Var[Wt] at horizon {}: {}'.format(horizon, EWt_dynamic - gamma*VarWt_dynamic))
    EWt_sbs = np.mean(W_T_Sbs)
    VarWt_sbs = np.var(W_T_Sbs)
    MeanVar_Sbs.append(EWt_sbs - gamma*VarWt_sbs)
    print ('Step by step E[Wt] - gamma*Var[Wt] at horizon {}: {}'.format(horizon, EWt_sbs - gamma*VarWt_sbs))

plt.scatter(timeList, MeanVar_Dynamic, label='dynamic')
plt.scatter(timeList, MeanVar_Sbs, label='step by step')
plt.title('Dynamic Vs Step by step Optimization')
plt.xlabel('Horizon')
plt.ylabel('MeanVar at horizon')
plt.legend(['dynamic', 'step by step'])
plt.savefig('Dynamic Vs Step by step Optimization')
plt.show()


