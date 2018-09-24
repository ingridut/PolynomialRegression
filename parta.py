from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import randrange, uniform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%matplotlib qt - Jupiter it lets the plot be plotted in separate window

#FrankeFunction - for simulation of data
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2)- 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(x.shape[0], x.shape[1])

#Ordinary Least Squared function
def ols(x, y, z, degree = 5):
    #x: vector of size(n, 1)
    #y: vector of size(n,1)
    # z: vector of size(n,1)
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform(xyb_)
    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
    return beta

#Mean Squared Error
def MSE(zReal, zPredicted):
    mse = np.mean((zReal-zPredicted)**2)
    return mse           

#Mean value of the function
def Mean(z):
    meanValue = np.mean(z)
    return meanValue

#R2 score function
def R2(zReal, zPredicted):
    meanValue = Mean(zReal)
    numerator = np.sum((zReal - zPredicted)**2)
    denominator = np.sum((zReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result

#Purpose of k-fold validation is to divide all the samples in k groups of samples equal sizes. 
#The prediction function is learned using k - 1 folds. We leave the last fold/subset for test.
def kfold(x, y, z, k=5):
    data_set = np.c_[x, y, z]
    np.random.shuffle(data_set)
    set_size = round(len(x)/k)
    folds=0

    MSE = []
    R2score = []
    betas = []

    while folds < len(x):
        # select variables in the test set
        test_indices = np.linspace(folds, folds+set_size, set_size)

        # training
        x_t = np.delete(data_set[:, 0], test_indices)
        y_t = np.delete(data_set[:, 1], test_indices)
        z_t = np.delete(data_set[:, 2], test_indices)

        # Ridge regression, save beta values
        beta = ols(x_t, y_t, z_t)
        betas.append(beta)

        # evaluation/test
        x_test = data_set[folds:folds+set_size, 0]
        y_test = data_set[folds:folds+set_size, 1]
        z_test = data_set[folds:folds+set_size, 2]

        # calculate the predicted z-values
        M_ = np.c_[x_test, y_test]
        poly = PolynomialFeatures(5)
        M = poly.fit_transform(M_)
        z_hat = M.dot(beta)

        # calculate MSE and R2scores
        MSE.append(MeanSquaredError(z_test, z_hat))
        R2score.append(R2(z_test, z_hat))

        folds += set_size

    # i = 1
    # print('k-fold validation with', len(MSE), 'folds')
    # for nr, test in enumerate(MSE):
    #     print('MSE for test nr', nr, '-->', test)

    return betas, MSE, R2score
###############################################################################
#All datat for test    
xx = np.load("data_for_part_1.npy")
x1 = xx[:,0].reshape(1000,1)
y1 = xx[:,1].reshape(1000,1)
z1 = FrankeFunction(x1, y1)
mseReturn, r2Return, zReturn, xY, zreal = kfold(x1, y1, z1)

