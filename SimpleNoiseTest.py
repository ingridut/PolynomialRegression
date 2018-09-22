import numpy as np
from RidgeRegression import RidgeRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from MSE import MeanSquaredError
import sklearn.linear_model as linear_model
from sklearn.preprocessing import PolynomialFeatures


def runTest(degree=5):

    # Make test data
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)

    z = FrankeFunction(x, y)
    print(np.shape(z))

    # Calculate beta
    beta = RidgeRegression(x, y, z, l=0)

    # Calculate y for a range of x-points
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    """
    Mnew = np.empty((len(x), rows))
    print(np.shape(Mnew))
    row = 0
    for i in range(degree+1):
        for j in range(degree+1):
            if i + j <= 5:
                Mnew[:, row] = np.power(x, i)*(np.power(y, j))
                row += 1
                """
    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)

    # Prediction
    Mnew = np.c_[np.ones((100**2, 1)), x, y, x ** 2, x * y, y ** 2,\
                 x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
                 x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
                 x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]
    predict = Mnew.dot(beta)

    # calculate MSE:
    z = FrankeFunction(x, y)
    MSE = MeanSquaredError(x, y, z, beta)
    print('MSE: ', MSE)

    # Plot the prediction
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_, y_, predict.reshape(100,100), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4)

def k_fold_validation(x, y, z, k=7):
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
        MSE.append(MeanSquaredError(x_test, y_test, z_test, beta))

        folds += set_size

    i = 1
    print('k-fold validation with', len(MSE), 'folds')
    for nr, test in enumerate(MSE):
        print('MSE for test nr', nr, '-->', test)

def bootstrap():
    pass

def scikitComparision():
    # Make data
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)

    z = FrankeFunction(x, y)

    X = np.c_[x, y]

    poly = PolynomialFeatures(degree=5)
    X_ = poly.fit_transform(X)
    #predict_ = poly.fit_transform(z)

    """
    M = np.c_[np.ones((len(x), 1)), x, y, x ** 2, x * y, y ** 2, \
              x ** 3, x ** 2 * y, x * y ** 2, y ** 3, \
              x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4, \
              x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]"""

    clf = linear_model.Ridge(alpha=0.1, fit_intercept=False)
    clf.fit(X_, z)
    beta = clf.coef_
    print(beta)

    beta2 = RidgeRegression(x, y, z)
    print(beta2)

    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)

    # Prediction
    Mnew = np.c_[np.ones((100**2, 1)), x, y, x ** 2, x * y, y ** 2,\
                 x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
                 x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
                 x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]
    predict = Mnew.dot(beta.T)

    # Plot the prediction
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_, y_, predict.reshape(100,100), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':
    scikitComparision()