from RidgeRegression import RidgeRegression
from Lasso import Lasso
from OrdinaryLeastSquare import ols
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from Analysis import MeanSquaredError, R2, bias, var_f



def k_fold_validation(x, y, z, k=5, method='Ridge'):
    data_set = np.c_[x, y, z]
    np.random.shuffle(data_set)
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

        if method == 'Ridge':
            # Ridge regression, save beta values
            beta = RidgeRegression(x_t, y_t, z_t)
            betas.append(beta)
        elif method == 'Lasso':
            beta = Lasso(x_t, y_t, z_t, degree=5)
        else:
            beta = ols(x_t, y_t, z_t, degree=5)

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

        folds += set_size

    M_MSE = np.mean(MSE)

    return M_MSE
