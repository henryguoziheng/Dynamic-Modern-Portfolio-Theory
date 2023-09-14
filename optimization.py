import numpy as np
import scipy as sp
from scipy.optimize import minimize

horizon = 2
p_list = np.array([[0.4, 0.5],
                   [0.6, 0.5]])
r_list = np.array([[-0.08, -0.07],
                   [0.11, 0.12]])
w0 = 10
r = 0.005
gamma = 0.5

def W1(phi):
    return w0*(1 + (1-phi[0])*r + phi[0]*r_list[:,0])

def W2(phi):
    x1 = W1(phi)
    y1 = 1 + (1-phi[1:])*r + phi[1:] * r_list[0, 1]
    x2 = W1(phi)
    y2 = 1 + (1-phi[1:])*r + phi[1:] * r_list[1, 1]
    temp = np.append(np.multiply(x1, y1), np.multiply(x2, y2))
    return temp

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

def MeanVar(phi):
    mean = np.dot(ProbTree(2)[:, 2], W2(phi))
    var = np.dot(ProbTree(2)[:, 2], np.square(W2(phi))) - np.power(mean, 2)
    return -(mean - gamma*var)

x0 = np.array([0, 0, 0])
res = minimize(MeanVar, x0, method='powell', options={'xtol': 1e-8, 'disp': True})
# With constraint of no short selling
# res = minimize(MeanVar, x0, method='SLSQP', constraints={'type': 'ineq', 'fun': lambda x: x}, options={'xtol': 1e-8, 'disp': True})

print(res)
# print (W1(phi))
# print (W2(phi))
# print (ProbTree(2))
# print (MeanVar(phi))