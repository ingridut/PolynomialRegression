from Analysis import plotFrankes, MeanSquaredError, R2, FrankeFunction, var2, varBeta, betaConfidenceInterval_OLS, bias, var_f
from OrdinaryLeastSquares import ols
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kFoldValidation import k_fold_validation
"""
    Analysis of a Ridge Regression model of Franke's function, using set of 1000 random x and y points
"""


# Load random data, 1000 points
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]

# Calculate Franke's function without noise
z = FrankeFunction(x, y, noise=0)

# Generate test data
x_test = np.random.rand(200)
y_test = np.random.rand(200)
z_test = FrankeFunction(x_test, y_test, noise=0)

#######################################################################################################################
# K-fold validation
# Further improve with k-fold validation
MSE = k_fold_validation(x, y, z, k=5, method='OLS')
print('MSE k-fold: ', MSE)

beta = ols(x, y, z, degree=5)

M_ = np.c_[x, y]
poly = PolynomialFeatures(5)
M = poly.fit_transform(M_)
z_hat = M.dot(beta)

# Calculate bias and variance
f_bias = bias(z, z_hat)
f_var = var_f(z_hat)

print('Bias: ', f_bias)
print('Variance: ', f_var)

# Calculate variance
conf1, conf2 = betaConfidenceInterval_OLS(z, beta, M)
print("BETA CONFIDENCE INTERVAL")
for i in range(len(beta)):
    print('Beta {0} = {1:.5f} & [{2:.5f}, {3:.5f}]'.format(i, beta[i], conf1[i], conf2[i]))