import time
import numpy as np
from numpy import power, prod, dot, cumprod, multiply, divide, square, exp, sqrt
from numpy.linalg import inv

p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],  # Probability for stock moving down or up for each step. Columns sum to 1.
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06], # Stock return at each time step corresponding to the probability matrix.
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])

def mean(p, r):
    temp = []
    for i in range(10):
        temp.append(dot(p[:, i], r[:, i]))
    return temp

def std(p, r):
    temp = []
    for i in range(10):
        temp1 = dot(square(r[:, i]), p[:, i])
        temp2 = square(dot(r[:, i], p[:, i]))
        temp.append(temp1 - temp2)
    return sqrt(temp)


print (mean(p_list, r_list))
print (std(p_list, r_list))