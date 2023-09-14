import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, transpose, ones
from numpy.linalg import inv, multi_dot
import matplotlib.pyplot as plt

assetNum = 1  # assume returns are independent
miu_dis = np.array([0.034])  # length is the same as asset numbers
sigma = np.array([0.09308061])  # std

w0 = 10  # W(0) starting wealth
r = 0.005  # risk-free rate
gamma = 0.5  # risk aversion rate. E(X)-gamma*Var(X)

def E(miu, sigma):
    return miu

def Var(miu, sigma):
    return square(sigma)


horizon = 2
# timeList = [1,2,3]
miuList = np.arange(0.01, 0.1, 0.01)

MeanVar_Dynamic = []
MeanVar_Sbs = []
for miu_dis[0] in miuList:

    def b_temp(t):  # if asset number > 1, then miu[t] change to miu[:, t]
        return square(E(miu_dis[0], sigma[0]) - r) / (Var(miu_dis[0], sigma[0]) + square(E(miu_dis[0], sigma[0]) - r))

    A = np.ones((3, horizon))
    for t in range(0, horizon):
        A[1, t] = (1 + r) * (1 - b_temp(t))
        A[2, t] = power(1 + r, 2) * (1 - b_temp(t))

    B = np.ones((3, horizon))
    for t in range(0, horizon):
        if t < horizon:
            B[1, t] = b_temp(t) * prod(A[1, t + 1:]) / (2 * prod(A[2, t + 1:]))
            B[2, t] = b_temp(t) * power(prod(A[1, t + 1:]) / (2 * prod(A[2, t + 1:])), 2)
        if t == horizon:
            B[1, t] = b_temp(t) * (1 / 2)
            B[2, t] = b_temp(t) * (1 / 4)

    miu = prod(A[1, :])
    temp2 = np.append(cumprod(A[1, 1:][::-1])[::-1], 1)
    nu = dot(temp2, B[1, :])
    tau = prod(A[2, :])
    a = nu / 2 - power(nu, 2)
    b = (miu * nu) / a
    c = tau - power(miu, 2) - a * power(b, 2)

    W_T_Dynamic = []
    W_T_Sbs = []
    pathNum = 10000
    for path in range(pathNum):
        simuReturnList = np.random.normal(miu_dis[0], sigma[0], horizon)
        #print (simuReturnList)

        W_Dynamic = [10]
        W_Sbs = [10]
        phi_Dynamic = []
        phi_Sbs = []
        for t in range(0, horizon):  # t = 0,1,2,3,...,9
            temp = (E(miu_dis[0], sigma[0]) - r) / (Var(miu_dis[0], sigma[0]) + square(E(miu_dis[0], sigma[0]) - r))
            if t < horizon - 1:
                phi_new_Dynamic = (-(1 + r) * W_Dynamic[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t + 1:horizon])) * temp) / W_Dynamic[t]  # find control at time t
                phi_Dynamic.append(phi_new_Dynamic)  # append control at time t
                W_new = multiply(W_Dynamic[t], 1 + (1-phi_Dynamic[t])*r + phi_Dynamic[t]*simuReturnList[t])
                W_Dynamic.append(W_new)
            if t == horizon - 1:
                phi_new_Dynamic = (-(1 + r) * W_Dynamic[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * temp)/W_Dynamic[t]
                phi_Dynamic.append(phi_new_Dynamic)  # append control at time t
                W_new = multiply(W_Dynamic[t], 1 + (1-phi_Dynamic[t])*r + phi_Dynamic[t]*simuReturnList[t])
                W_Dynamic.append(W_new)

            phi_new_Sbs = (E(miu_dis[0], sigma[0]) - r) / (2 * gamma * W_Sbs[t] * Var(miu_dis[0], sigma[0]))
            phi_Sbs.append(phi_new_Sbs)
            W_new_Sbs = multiply(W_Sbs[t], 1 + (1-phi_Sbs[t])*r + phi_Sbs[t]*simuReturnList[t])
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

plt.scatter(miuList, MeanVar_Dynamic, label='dynamic')
plt.scatter(miuList, MeanVar_Sbs, label='step by step')
plt.title('Dynamic Vs Step by step Optimization varying asset mean return')
plt.xlabel('Mean Return')
plt.ylabel('MeanVar at horizon T=2')
plt.legend(['dynamic', 'step by step'])
plt.savefig('Dynamic Vs Step by step Optimization varying asset mean return')
plt.show()