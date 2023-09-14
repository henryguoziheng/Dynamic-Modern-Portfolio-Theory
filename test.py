import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, transpose

alpha = np.transpose(np.array([0.5, 0.3, 0.2]))
sigma = np.array([[1.5, 0, 0],
                  [0, 0.9, 0],
                  [0, 0, 1.2]])
one = np.ones((3,1))

gamma = 0.5

a = np.linalg.multi_dot([transpose(alpha), sigma, alpha])
b = np.linalg.multi_dot([transpose(alpha), sigma, one])
c = np.linalg.multi_dot([transpose(one), sigma, one])

d = (a)/(2*gamma) + ((2*gamma-b)*b)/(2*gamma*c)


print (a/(4*gamma**2) + ((2*gamma-b)*b)/(2*power(gamma, 2)*c) + power(2*gamma-b, 2)/(4*power(gamma, 2)*c))
print ((c*power(d,2) - 2*b*d + a)/(a*c - power(b,2)))