import time
import numpy as np
from numpy.linalg import multi_dot, inv
from numpy import power, prod, dot, cumprod, multiply, divide, reshape, square, ones, outer, transpose

start = time.time()
# parameters
horizon = 1  # Toal time range for wealth management
assetNum = 2
p1_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r1_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
p2_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r2_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate

def b_temp(t):
    temp_a = ones((assetNum, 1))
    temp_a[0] = dot(reshape(r1_list[:,t] - r * np.ones(2), (1,2)), p1_list[:,t])
    temp_a[1] = dot(reshape(r2_list[:,t] - r * np.ones(2), (1,2)), p2_list[:,t])
    temp_b =ones((assetNum, assetNum))
    temp_b[0,0] = dot(square(reshape(r1_list[:,t] - r * ones(2), (1,2))), p1_list[:,t])
    temp_b[1,1] = dot(square(reshape(r2_list[:,t] - r * ones(2), (1,2))), p2_list[:,t])
    temp_b[0,1] = dot(outer(r1_list[:,t], r2_list[:,t]).flatten(), outer(p1_list[:,t], p2_list[:,t]).flatten()) - dot(r1_list[:,t], p1_list[:,t]) - dot(r2_list[:,t],p2_list[:,t]) - square(r)
    temp_b[1,0] = temp_b[0,1]
    temp_c = ones((assetNum, 1))
    temp_c[0] = dot(reshape(r1_list[:,t] - r * np.ones(2), (1,2)), p1_list[:,t])
    temp_c[1] = dot(reshape(r2_list[:,t] - r * np.ones(2), (1,2)), p2_list[:,t])
    return multi_dot([transpose(temp_a), inv(temp_b), temp_c])


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
pathNum = 5
for path in range(pathNum):
    slctVec1 = np.random.binomial(1, p1_list[1, 0:horizon])
    r_list_new1 = np.transpose(r1_list[:, 0:horizon])
    slctVec2 = np.random.binomial(1, p2_list[1, 0:horizon])
    r_list_new2 = np.transpose(r2_list[:, 0:horizon])
    simuReturnList1 = r_list_new1[np.arange(len(r_list_new1)), slctVec1]
    simuReturnList2 = r_list_new2[np.arange(len(r_list_new2)), slctVec2]
    simuReturnList = ones((assetNum, horizon))
    simuReturnList[0, :] = simuReturnList1
    simuReturnList[1, :] = simuReturnList2
    # print (simuReturnList)

    W = [10]
    phi = []
    for t in range(0, horizon): # t = 0,1,2,3,...,9

        temp_b = ones((assetNum, assetNum))
        temp_b[0, 0] = dot(square(reshape(r1_list[:, t] - r * ones(2), (1, 2))), p1_list[:, t])
        temp_b[1, 1] = dot(square(reshape(r2_list[:, t] - r * ones(2), (1, 2))), p2_list[:, t])
        temp_b[0, 1] = dot(outer(r1_list[:, t], r2_list[:, t]).flatten(),
                           outer(p1_list[:, t], p2_list[:, t]).flatten()) - dot(r1_list[:, t], p1_list[:, t]) - dot(
            r2_list[:, t], p2_list[:, t]) - square(r)
        temp_b[1, 0] = temp_b[0, 1]
        temp_c = ones((assetNum, 1))
        temp_c[0] = dot(reshape(r1_list[:, t] - r * np.ones(2), (1, 2)), p1_list[:, t])
        temp_c[1] = dot(reshape(r2_list[:, t] - r * np.ones(2), (1, 2)), p2_list[:, t])
        temp = dot(temp_b, temp_c)

        if t < horizon-1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t+1:horizon])) * temp) / W[t]
            phi.append(phi_new)
            W_new = multiply(W[t], 1 + (1 - dot(ones((1,2)), phi[t]))*r + dot(transpose(phi[t]), simuReturnList[:,t]))
            W.append(W_new)
        if t == horizon-1:
            phi_new = (-(1 + r) * W[t] * temp + 0.5 * (b * 10 + (nu / (2 * gamma * a))) * temp) / W[t]
            phi.append(phi_new)
            W_new = multiply(W[t], 1 + (1 - dot(ones((1,2)), phi[t]))*r + dot(transpose(phi[t]), simuReturnList[:,t]))
            W.append(W_new)
    #print(phi)
    W_T.append(W[-1])

EWt = np.mean(W_T)
VarWt = np.var(W_T)
print ('E[Wt] - gamma*Var[Wt]: {}'.format(EWt - gamma*VarWt))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))
