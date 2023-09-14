import numpy as np
from numpy.linalg import multi_dot
from numpy import dot, transpose
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import matplotlib.pylab as plt

alpha = np.array([0.3, 0.1, 0.05])
sigma = np.array([[np.square(0.35), 0.35*0.15, 0.35*0.05],
                  [0.35*0.15, np.square(0.15), 0.15*0.05],
                  [0.35*0.05, 0.15*0.05, np.square(0.05)]])

gammaList = np.arange(0.1, 10, 0.1)

gammaPortfolioMean = []
gammaPortfolioVar = []
for gamma in gammaList:

    def meanVar(phi):
        return -(dot(transpose(phi), alpha) - gamma * multi_dot([transpose(phi), sigma, phi]))

    gammaBounds = Bounds([0, 0, 0], [np.inf, np.inf, np.inf])
    gamma_linear_constraint = LinearConstraint([1, 1, 1], [1], [1])

    phi0 = np.array([0.5, 0.5, 0])

    gamma_res = minimize(meanVar, phi0, method='SLSQP', constraints=[gamma_linear_constraint], bounds=gammaBounds)
    gamma_phi_star = gamma_res.x
    gammaPortfolioMean.append(dot(transpose(alpha), gamma_phi_star))
    gammaPortfolioVar.append(multi_dot([transpose(gamma_phi_star), sigma, gamma_phi_star]))



epsilonList = np.arange(0.05, 0.3, 0.01)

epsilonPortfolioMean = []
epsilonPortfolioVar = []
for epsilon in epsilonList:

    def minimizeVar(phi):
        return multi_dot([transpose(phi), sigma, phi])

    epsilonBounds = Bounds([0, 0, 0], [np.inf, np.inf, np.inf])
    epsilon_linear_constraint = LinearConstraint([[1, 1, 1], transpose(alpha)], [1, epsilon], [1, epsilon])

    phi_0 = np.array([0.5, 0.5, 0])
    epsilon_res = minimize(minimizeVar, phi_0, method='SLSQP', constraints=[epsilon_linear_constraint], bounds=epsilonBounds)
    epsilon_phi_star = epsilon_res.x
    epsilonPortfolioMean.append(dot(transpose(alpha), epsilon_phi_star))
    epsilonPortfolioVar.append(multi_dot([transpose(epsilon_phi_star), sigma, epsilon_phi_star]))


plt.scatter(gammaPortfolioVar, gammaPortfolioMean)
plt.scatter(epsilonPortfolioVar, epsilonPortfolioMean)
plt.title('Efficient Frontier (No Short Selling)')
plt.xlabel('Variance')
plt.ylabel('Mean')
plt.legend(['Max MeanVar', 'Min Variance'])
plt.show()

