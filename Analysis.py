"""
Ridge regression analysis of Franke's function
"""

from RidgeRegression import RidgeRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def FrankeFunction(x,y, noise=0.1):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4 + noise*np.random.randn(len(x), 1))

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

def betaConfInt(beta, X, var2, alpha=0.025):
    """
    Comput a 1-2*alpha confidence interval for the beta values
    :param beta_co_mat:
    :param alpha:
    :return:
    """
    v = np.diag(np.linalg.inv(X.T.dot(X)))
    i_minus = beta-v*1.96*np.sqrt(var2)
    i_plus = beta+v*1.96*np.sqrt(var2)

    return i_minus, i_plus

def varBeta(X, var2):
    var = np.linalg.inv(X.T.dot(X))*(var2)
    return var

def var2(z, z_hat, p):
    nom = np.square(z-z_hat)
    return np.sum(nom)/(len(z)-p-1)

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

def plotFrankes(num_of_points, beta, degree=5):

    x = np.linspace(0, 1, num_of_points)
    y = np.arange(0, 1, num_of_points)

    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)

    M = np.c_[x, y]
    poly = PolynomialFeatures(degree=5)
    M_ = poly.fit_transform(M)
    predict = M_.dot(beta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_, y_, predict.reshape(100,100), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
