from Analysis import plotFrankes, MeanSquaredError, R2, FrankeFunction, var2, varBeta, betaConfidenceInterval_OLS, bias, var_f
from OrdinaryLeastSquare import ols
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from kFoldValidation import k_fold_validation



# Evaluate model with bootstrap
X = np.load('data_for_part_1.npy')
x = X[:, 0]
y = X[:, 1]
z = FrankeFunction(x, y, noise=0.1)

MSE, R2_b, bias, variance = bootstrap(x, y, z, method='OLS', p_degree=5)
print('--- BOOTSTRAP ---')
print('MSE: ', MSE)
print('R2: ', R2_b)
print('Bias: ', bias)
print('Variance: ', variance)
