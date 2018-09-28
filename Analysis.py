"""
Ridge regression analysis of Franke's function
"""

from RidgeRegression import RidgeRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def FrankeFunction(x,y, noise=0.01):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4 + noise*np.random.randn(len(x)))

def R2(zReal, zPredicted):
    """
    :param zReal: actual z-values, size (n, 1)
    :param zPredicted: predicted z-values, size (n, 1)
    :return: R2-score
    """
    meanValue = np.mean(zReal)
    numerator = np.sum((zReal - zPredicted)**2)
    denominator = np.sum((zReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result

def MeanSquaredError(z, z_hat):
    """
    :param z: actual z-values, size (n, 1)
    :param z_hat: predicted z-values, size (n, 1)
    :return: Mean squared error
    """
    MSE = np.square(z-z_hat).mean()
    return MSE


def betaConfidenceInterval_Ridge(z_real, beta, X, l):
    """
    Compute a 90% confidence interval for the beta coefficients - Ridge
    """

    # Calculate variance squared in the error
    z_hat = X.dot(beta)
    N, P = np.shape(X)
    sigma_2 = (np.sum(np.power((z_real-z_hat), 2)))/N

    # Calculate the variance squared of the beta coefficients
    XTX= X.T.dot(X)
    R, R = np.shape(XTX)
    var_beta = np.diag(sigma_2*np.linalg.inv((XTX + l*np.identity(R))))

    # The square root of var_beta is the standard error. Use it to calculate confidence intervals
    i_minus = beta - 1.645*np.sqrt(var_beta/N)
    i_plus = beta + 1.645*np.sqrt(var_beta/N)

    return i_minus, i_plus

def betaConfidenceInterval_OLS(z_real, beta, X):
    """
    Compute a 90% confidence interval for the beta coefficients
    """

    # Calculate variance squared in the error
    z_hat = X.dot(beta)
    N, P = np.shape(X)
    sigma_2 = (np.sum(np.power((z_real-z_hat), 2)))/N

    # Calculate the variance squared of the beta coefficients
    var_beta = np.diag(sigma_2*np.linalg.inv((X.T.dot(X))))

    # The square root of var_beta is the standard error. Use it to calculate confidence intervals
    i_minus = beta - 1.645*np.sqrt(var_beta/N)
    i_plus = beta + 1.645*np.sqrt(var_beta/N)

    return i_minus, i_plus

def varBeta(X, var2):
    """
    Computes the variance of the beta values
    :param X: polynomial matrix
    :param var2: variance squared
    :return: variance of beta
    """
    var = np.linalg.inv(X.T.dot(X))*(var2)
    return var

def var2(z, z_hat, p=21):
    """
    Computes the variance
    :param z: real z-values
    :param z_hat: model z-values
    :param p: degree of polynomial fit
    :return: variance squared
    """
    nom = np.square(z-z_hat)
    return np.sum(nom)/(len(z)-p-1)

def bias(z, z_hat):
    """
    Computes the bias in the model
    :param z: real z-values
    :param z_hat: model z-values
    :return: bias squared
    """
    z_stroke = np.mean(z_hat)
    return (np.square(z-z_stroke)).mean()

def var_f(z_hat):
    """
    Computes the variance of the model
    :param z_hat: model z-values
    :return: variance
    """
    z_stroke = np.mean(z_hat)
    return np.square(z_hat-z_stroke).mean()

def k_fold_validation(x, y, z, k=5):
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
        beta = RidgeRegression(x_t, y_t, z_t)
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

def plotFrankes(beta, degree=5):

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)

    M = np.c_[x, y]
    poly = PolynomialFeatures(degree=degree)
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_, y_, predict.reshape(100, 100), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
