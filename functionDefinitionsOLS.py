
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%matplotlib qt

#FrankeFunction - for simulation of data
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2)- 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(x.shape[0], x.shape[1])
    
#Ordinary Least Squared function
def ols(x, y, z, degree = 2):
    #x: vector of size(n, 1)
    #y: vector of size(n,1)
    # z: vector of size(n,1)
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform(xyb_)
    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
    return beta

#Mean Squared Error
def mse(zReal, zPredicted):
    mse = np.mean((zReal-zPredicted)**2)
    return mse           

def bias(zReal, zPredicted):
    bias = np.mean( (zReal- np.mean(zPredicted))**2 )
    return bias
    
def var(zReal, zPredicted):
    var = np.mean( (zPredicted - np.mean(zPredicted))**2 )
    return var

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

        # OLS regression, save beta values
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
        MSE.append(mse(z_test, z_hat))
        R2score.append(R2(z_test, z_hat))

        folds += set_size

    # i = 1
    # print('k-fold validation with', len(MSE), 'folds')
    # for nr, test in enumerate(MSE):
    #     print('MSE for test nr', nr, '-->', test)

    return betas, MSE, R2score