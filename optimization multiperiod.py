import time
import numpy as np
from numpy import power
import scipy as sp
from scipy.optimize import minimize

start = time.time()

# parameters
horizon = 1 # Total time range for wealth management
p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10 # W(0) starting wealth
gamma = 0.5 # risk aversion rate. E(X)-gamma*Var(X)
r = 0.005 # risk-free rate


def W(phi, t): # input phi is a (2^(t+1)-1) array
    '''
    Wealth process.
    :param phi: (2^(t+1)-1) array. weight of stock at each time period. e.g phi[0] is holding at 0, phi[1:3] holding at 1, phi[3:7] holding at 2 ...
    :param t: int. time.
    :return: wealth at time t under given strategy from 0 to t.
    '''
    if t == 0:
        return w0
    if t >= 1:
        x1 = W(phi ,t-1)
        y1 = (1 + (1-phi[power(2,t-1)-1:power(2,t)-1])*r + phi[power(2,t-1)-1:power(2,t)-1]*r_list[0, t-1])
        x2 = W(phi ,t-1)
        y2 = (1 + (1-phi[power(2,t-1)-1:power(2,t)-1])*r + phi[power(2,t-1)-1:power(2,t)-1]*r_list[1, t-1])
        temp = np.append(np.multiply(x1,y1), np.multiply(x2,y2))
        return temp

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

def MeanVar(phi, t):
    '''
    Objective function. Calculates E(Xt) - gamma*Var(Xt) for given time t and strategy phi.
    :param phi: (2^(t+1)-1) array. weight of stock at each time period. e.g phi[0] is holding at 0, phi[1:3] holding at 1, phi[3:7] holding at 2 ...
    :param t: int. time.
    :return: float. E(Xt) - gamma*Var(Xt) for given time t and strategy phi.
    '''
    mean = np.dot(ProbTree(t)[0:2**t, t], W(phi, t))
    var = np.dot(ProbTree(t)[0:2**t, t], np.square(W(phi, t))) - np.power(mean, 2)
    return -(mean - gamma*var)

# Start doing optimization
phi = np.ones((power(2,horizon)-1, 1)) # initial guess
#res = minimize(MeanVar, phi, args=(horizon),method='powell', options={'xtol': 1e-6, 'disp': True}) # without constraint
res = minimize(MeanVar, phi,args=(horizon), method='SLSQP', constraints={'type': 'ineq', 'fun': lambda x: x}, options={'xtol': 1e-8, 'disp': True}) # With constraint of no short selling


print (res)
end = time.time()
print ('Time elapsed: {} seconds'.format(end - start))
