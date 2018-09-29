from RidgeRegression import RidgeRegression
from Lasso import Lasso
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from Analysis import MeanSquaredError, R2, bias, var_f


def k_fold_validation(x, y, z, k=5, method='Ridge'):
    data_set = np.c_[x, y, z]
    np.random.shuffle(data_set)
    set_size = round(len(x)/k)
    folds=0

    MSE = []
    R2score = []
    f_bias = []
    f_var = []
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
            #OLS goes here
            pass

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
        f_bias.append(bias(z_test, z_hat))
        f_var.append(var_f(z_hat))
        folds += set_size

    return betas, MSE, R2score, f_bias, f_var