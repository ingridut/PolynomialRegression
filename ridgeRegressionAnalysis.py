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

def betaConfInt(beta, z, X, var2, alpha=0.025):
    """
    Comput a 1-2*alpha confidence interval for the beta values
    :param beta_co_mat:
    :param alpha:
    :return:
    """
    v = np.linalg.inv(X.T.dot(X))
    i_minus = beta-np.power(z, 1-alpha).dot(v)*np.sqrt(var2)
    i_plus = beta+np.power(z, 1-alpha).dot(v)*np.sqrt(var2)

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

    # calculate the predicted z-values
    M_ = np.c_[x, y]
    poly3 = PolynomialFeatures(3)
    M = poly3.fit_transform(M_)
    zpredict_3 = M.dot(beta_3)

    poly4 = PolynomialFeatures(4)
    M = poly4.fit_transform(M_)
    zpredict_4 = M.dot(beta_4)

    poly5 = PolynomialFeatures(5)
    M = poly5.fit_transform(M_)
    zpredict_5 = M.dot(beta_5)

    # Calculate beta variance



    # Choose optimal MSE, R2-score
    print('=== INVESTIGATE DEGREES ===')
    MSE_3 = MeanSquaredError(z, zpredict_3)
    R2_3 = R2(z, zpredict_3)
    print('--- Degrees: 3 ---\n Mean Squared error: {0:.7f} \n R2 Score: {1:.7f}\n'.format(MSE_3, R2_3))
    MSE_4 = MeanSquaredError(z, zpredict_4)
    R2_4 = R2(z, zpredict_4)
    print('--- Degrees: 4 ---\n Mean Squared error: {0:.7f} \n R2 Score: {1:.7f}\n'.format(MSE_4, R2_4))
    MSE_5 = MeanSquaredError(z, zpredict_5)
    R2_5 = R2(z, zpredict_5)
    print('--- Degrees: 5 ---\n Mean Squared error: {0:.7f} \n R2 Score: {1:.7f}\n'.format(MSE_4, R2_4))

    # calculate beta values with various lambdas
    lambdas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    betas = []
    for la in lambdas:
        betas.append(RidgeRegression(x, y, z, 5, l=la))

    # Choose optimal MSE, R2-score
    poly5 = PolynomialFeatures(5)
    M = poly5.fit_transform(M_)
    MSEs = []
    R2s = []
    for b in betas:
        zpredict = M.dot(b)
        MSEs.append(MeanSquaredError(z, zpredict))
        R2s.append(R2(z, zpredict))

    print('INVESTIGATE LAMBDA VALUES')
    for i in range(len(betas)):
        print('--- Lambda value: {0} ---\n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(lambdas[i],
                                                                                                      MSEs[i], R2s[i]))

    # Further improve with k-fold validation
    betas, MSEs, R2scores = k_fold_validation(x, y, z, k=5)
    print('K-FOLD VALIDATION ')
    for i in range(5):
        print('--- Fold nr: {0} ---\n Mean Squared error: {1:.7f} \n R2 Score: {2:.7f}\n'.format(i+1, MSEs[i], R2scores[i]))


    # choose optimal MSE,

