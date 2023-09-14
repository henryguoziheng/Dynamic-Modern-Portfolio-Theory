import numpy as np
from numpy import power, dot, square

horizon = 2
p_list = np.array([[0.4, 0.5],
                   [0.6, 0.5]])
r_list = np.array([[-0.01, -0.02], [0.01, 0.025]])
w0 = 10
gamma = 0.5
r = 0.005


def W(t):
    if t == 0:
        return np.array([10])
    if t >= 1:
        x = W(t-1)
        y = (1 + (1-phi(t-1))*r + phi(t-1)*r_list[:, t-1])
        temp = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        return np.multiply(temp[:,0], temp[:, 1])

def phi(t):
    Er1s = dot(r_list[:,0], p_list[:, 0])
    Er1s_sqr = dot(square(r_list[:,0]), p_list[:, 0])
    Er2s = dot(r_list[:,1], p_list[:, 1])
    Er2s_sqr = dot(square(r_list[:,1]), p_list[:, 1])
    x1 = square(r_list[:,0])
    y1 = r_list[:,1]
    temp1 = np.dstack(np.meshgrid(x1, y1)).reshape(-1, 2)
    Er1s_sqr_r2s = dot(np.multiply(temp1[:,0], temp1[:, 1]), ProbTree(2)[:,2])
    x2 = r_list[:,0]
    y2 = square(r_list[:,1])
    temp2 = np.dstack(np.meshgrid(x2, y2)).reshape(-1, 2)
    Er1s_r2s_sqr = dot(np.multiply(temp2[:,0], temp2[:, 1]), ProbTree(2)[:,2])
    return (Er1s+Er2s+Er1s*Er2s-W(0)*(2*r+power(r, 2)))/(2*gamma*(Er1s_sqr+Er2s_sqr+Er1s_sqr*Er2s_sqr+2*Er1s*Er2s+2*Er1s_sqr_r2s+2*Er1s_r2s_sqr-power(Er1s+Er2s+Er1s*Er2s,2)))

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
print (WealthPath(2))
print (ProbTree(2))
print (MeanVar(2))