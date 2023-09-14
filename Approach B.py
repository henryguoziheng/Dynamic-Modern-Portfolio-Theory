import numpy as np
from numpy import power, dot, square, multiply

horizon = 5
p_list = np.array([[0.4, 0.5, 0.2, 0.55, 0.5, 0.7, 0.45, 0.5, 0.7, 0.45],
                   [0.6, 0.5, 0.8, 0.45, 0.5, 0.3, 0.55, 0.5, 0.3, 0.55]])
r_list = np.array([[-0.08, -0.12, -0.09, -0.07, -0.06, -0.13, -0.05, -0.08, -0.05, -0.06],
                   [0.11, 0.09, 0.12, 0.08, 0.035, 0.10, 0.06, 0.03, 0.06, 0.09]])
w0 = 10
gamma = 0.5
r = 0.005


def W(t):
    if t == 0:
        return np.array([10])
    if t >= 1:
        x1 = W(t-1)
        y1 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[0, t-1])
        x2 = W(t-1)
        y2 = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[1, t-1])
        temp = np.append(multiply(x1,y1), multiply(x2,y2))
        return temp

def phi(t):
    def Er(t):
        return dot(r_list[:, t], p_list[:, t])
    def ErSqr(t):
        return dot(square(r_list[:,t]), p_list[:, t])
    return (W(t)*(Er(t)-r))/(2*gamma*power(W(t), 2)*(ErSqr(t) - power(Er(t), 2)))


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
print (phi(2))
print (phi(3))
print (phi(horizon-1))
print (MeanVar(horizon))