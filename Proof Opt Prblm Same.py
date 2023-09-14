import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, transpose
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


alpha = np.transpose(np.array([0.5, 0.3, 0.2]))
sigma = np.array([[1.5, 0, 0],
                  [0, 0.9, 0],
                  [0, 0, 1.2]])
one = np.ones((3,1))

gamma = 0.5

def meanVar(phi):
    '''
    :param phi: column vector
    :return:
    '''
    mean = dot(transpose(phi), alpha)
    var = np.linalg.multi_dot([transpose(phi), sigma, phi])
    return -(mean - gamma*var)

phi = np.array([1, 0, 0]).T
cons = ({'type': 'ineq', 'fun': lambda x: x}, {'type': 'eq', 'fun': lambda x: sum(x) - 1})
res = minimize(meanVar, phi, method='SLSQP', constraints=cons, options={'disp': False}) # With constraint of no short selling

phi_star = res.x

d = dot(phi_star, alpha)
d2 = np.linalg.multi_dot([transpose(phi_star), sigma, phi_star])
print (d, d2)


def minVar(phi2):
    return np.linalg.multi_dot([transpose(phi2), sigma, phi2])

phi2 = np.array([1, 0, 0]).T
cons2 = ({'type': 'ineq', 'fun': lambda x: x},
         {'type': 'eq', 'fun': lambda x: sum(x) - 1},
         {'type': 'eq', 'fun': lambda x: dot(x, alpha) - d})
res2 = minimize(minVar, phi, method='SLSQP', constraints=cons2, options={'disp': True}) # With constraint of no short selling
print (res2)

