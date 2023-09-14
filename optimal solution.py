import numpy as np
from numpy import transpose, dot, array
from numpy.linalg import multi_dot, inv
import matplotlib.pyplot as plt
import yfinance as yf

# data = yf.download("SPY AAPL TSLA USO", start="2020-08-30", end="2021-08-30")['Adj Close']
# print (data.cov())

gammaList = np.arange(0.001, 0.1, 0.001)
epsilonList = []
for gamma in gammaList:

    alpha = np.array([0.04, 0.10, 0.15])
    covariance = np.array([[0.11**2, 0, 0],
                           [0, 0.21**2, 0],
                           [0, 0, 0.4**2]])
    one = np.ones((3))

    #gamma = 0.5

    a = multi_dot([transpose(alpha), inv(covariance), alpha])
    b = multi_dot([transpose(alpha), inv(covariance), one])
    c = multi_dot([transpose(one), inv(covariance), one])

    phi_gamma = 0.5*(1/gamma)*(dot(inv(covariance), alpha) - ((b-2*gamma)/c)*dot(inv(covariance), one))

    eplison = dot(transpose(alpha), phi_gamma)

    temp1 = transpose(array([dot(inv(covariance), alpha), dot(inv(covariance), one)]))
    temp2 = inv(array([[a, b],
                       [b, c]]))
    temp3 = array([[eplison],
                   [1]])

    phi_epsilon = multi_dot([temp1, temp2, temp3])

    epsilonList.append(eplison)
    print (phi_gamma)
    # print (phi_epsilon)
    # print (eplison)

plt.plot(gammaList, epsilonList)
plt.show()