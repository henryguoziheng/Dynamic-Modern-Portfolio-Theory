import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide

start = time.time()
# parameters
horizon = 3 # Toal time range for wealth management
p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate

# intermediate steps for finding dynamic optimization results according to Duan Li (2000)
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

def W(t):
    '''
    Wealth process value at time t
    :param t: int, time
    :return: array, lenght (2^t * 1)
    '''
    if t == 0:
        return np.array([w0])
    if t >= 1:
        x1 = W(t-1)
        y1 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[0, t-1])
        x2 = W(t-1)
        y2 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[1, t-1])
        temp = np.append(multiply(x1,y1), multiply(x2,y2))
        return temp

def phi(t):
    '''
    Strategy or control. Weight of stock hold at time t
    :param t: int, time
    :return: array, lenght (2^t * 1)
    '''
    temp = np.dot(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2)), p_list[:, t]) / np.dot(np.square(np.reshape(r_list[:, t] - r * np.ones(2), (1, 2))), p_list[:, t])
    if t < horizon - 1:
        return (-(1 + r) * W(t) * temp + 0.5 * (b * W(0) + (nu / (2 * gamma * a))) * (prod(divide(A[1, :], A[2, :])[t+1:horizon])) * temp) / W(t)
    if t == horizon - 1:
        return (-(1 + r) * W(t) * temp + 0.5 * (b * W(0) + (nu / (2 * gamma * a))) * temp) / W(t)

def WealthPath(t):
    '''
    Stores each wealth path from t=0 to t
    :param t: int, time
    :return: array, (2^t * t)
    '''
    if t == 0:
        return W(0)
    if t >= 1:
        mat = np.zeros((2**t, t+1))
        for i in range(t+1):
            mat[0:2**i, i] = W(i)
        return mat

def ProbTree(t):
    '''
    Stores probability at each node. Probability from t=0 to this node.
    :param t: int. time.
    :return: (2^t * t) array.
    '''
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
    '''
    Objective function. our problem is to find pht(t) s.t. max(E[Wt] - gamma*Var(Wt))
    :param t: int, time
    :return: float, E[Wt] - gamma*Var(Wt)
    '''
    if t == 0:
        return W(0)
    if t >= 1:
        mean = np.dot(np.transpose(ProbTree(t)[0:2**t, t]), W(t))
        var = np.dot(np.transpose(ProbTree(t)[0:2**t, t]), np.square(W(t))) - mean**2
        return mean - gamma*var


print (phi(0))
print (WealthPath(horizon))
# print (ProbTree(4))
# print (MeanVar(horizon))
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))