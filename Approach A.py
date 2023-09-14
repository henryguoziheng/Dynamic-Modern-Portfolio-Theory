import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide

horizon = 2
p_list = np.array([[0.4, 0.5],
                   [0.6, 0.5]])
r_list = np.array([[-0.08, -0.07],
                   [0.11, 0.12]])
w0 = 10
gamma = 0.5
r = 0.005

def b_temp(t): # change to array instead of function later
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

def W(t):
    if t == 0:
        return np.array([w0])
    if t >= 1:
        x1 = W(t-1)
        y1 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[0, t-1])
        x2 = W(t-1)
        y2 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[1, t-1])
        temp = np.append(multiply(x1,y1), multiply(x2,y2))
        return temp

def phi(t): #for one stock one MMA
    temp = np.dot(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2)), p_list[:, t]) / np.dot(np.square(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2))), p_list[:, t])
    if t < horizon - 1:
        return (-(1 + r) * W(t) * temp + 0.5 * (b * W(0) + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t+1:horizon])) * temp) / W(t)
    if t == horizon - 1:
        return (-(1 + r) * W(t) * temp + 0.5 * (b * W(0) + (nu / (2 * gamma * a))) * temp) / W(t)

def WealthPath(t):
    if t == 0:
        return W(0)
    if t >= 1:
        mat = np.zeros((2**t, t+1))
        for i in range(t+1):
            mat[0:2**i, i] = W(i)
        return mat

def ProbTree(t):
    mat = np.zeros((2**t, t + 1))
    mat[0,0] = 1
    if t== 0:
        mat[0,0] = 1
        return mat
    if t >= 1:
        for i in range(1,t+1):
            x = ProbTree(i-1)[0:2 ** (i-1), i-1]
            y = p_list[:, i-1]
            temp = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
            mat[0:2 ** i, i] = np.multiply(temp[:,0], temp[:, 1])
        return mat

def MeanVar(t):
    if t == 0:
        return W(0)
    if t >= 1:
        mean = np.dot(np.transpose(ProbTree(t)[0:2**t, t]), WealthPath(t)[0:2**t, t])
        var = np.dot(np.transpose(ProbTree(t)[0:2**t, t]), np.square(WealthPath(t)[0:2**t, t])) - mean**2
        return mean - gamma*var

print (phi(0))
print (phi(1))
print (WealthPath(2))
print (MeanVar(2))
