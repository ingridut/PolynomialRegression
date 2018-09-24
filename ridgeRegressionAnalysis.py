"""
Ridge regression analysis of Franke's function
"""

from RidgeRegression import RidgeRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4)

def R2Score():
    pass

def MeanSquaredError(x, y, z, degrees, beta):
    """
    Calculates the Mean Squared Error
    :param y: numpy vector with y data, size (n, 1)
    :param x: numpy vector with x data, size (n, 1)
    :param beta: model
    :return: Mean squared error
    """
    # Calculate z_hat, the predicted z-values
    M_ = np.c_[x, y]
    poly = PolynomialFeatures(degrees)
    M = poly.fit_transform(M_)
    z_hat = M.dot(beta)

    # Calculate MSE
    MSE = 0
    for i in range(0, len(z)):
        MSE += np.power(z[i] - z_hat[i], 2)
    MSE = MSE/len(z)

    return MSE

def k_fold_validation(x, y, z, k=5):
    data_set = np.c_[x, y, z]
    np.random.shuffle(data_set)
    print(np.shape(data_set))
    set_size = round(len(x)/k)
    folds=0

    MSE = []
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
        MSE.append(MeanSquaredError(x_test, y_test, z_test, degrees=5, beta=beta))

        folds += set_size

    i = 1
    print('k-fold validation with', len(MSE), 'folds')
    for nr, test in enumerate(MSE):
        print('MSE for test nr', nr, '-->', test)

if __name__ == "__main__":
    # Load random data, 1000 points
    X = np.load('data_for_part_1.npy')
    x = X[:, 0]
    y = X[:, 1]

    # Compute Franke's function
    z = FrankeFunction(x, y)

    # calculate beta values with various degrees
    beta_3 = RidgeRegression(x, y, z, 3, l=0)
    beta_4 = RidgeRegression(x, y, z, 4, l=0)
    beta_5 = RidgeRegression(x, y, z, 5, l=0)

    # Choose optimal MSE, R2-score
    MSE_3 = MeanSquaredError(x, y, z, beta=beta_3, degrees=3)
    print(MSE_3)
    MSE_4 = MeanSquaredError(x, y, z, beta=beta_4, degrees=4)
    print(MSE_4)
    MSE_5 = MeanSquaredError(x, y, z, beta=beta_5, degrees=5)
    print(MSE_5)

    # calculate beta values with various lambdas
    lambdas = [0.2, 0.4, 0.6, 0.8, 1]
    betas = [beta_5]
    for la in lambdas:
        betas.append(RidgeRegression(x, y, z, 5, l=la))

    # Choose optimal MSE, R2-score
    MSEs = []
    for b in betas:
        MSEs.append(MeanSquaredError(x, y, z, degrees=5, beta=b))

    print(MSEs)

    # Further improve with k-fold validation
    k_fold_validation(x, y, z, k=5)

    # choose optimal MSE,

