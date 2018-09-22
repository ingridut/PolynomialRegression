import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def RidgeRegression(x, y, z, degree=5, l=0.1):
    """
    Linear regression using Ridge method
    :param x: numpy vector of size (n, 1)
    :param y: numpy vector of size (n, 1)
    :param degree: degree of polynomial fit
    :param l: Ridge
    :return: numpy array, size (rows, 1)

    """
    M_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    M = poly.fit_transform(M_)
    # Calculate matrix with powers of x and y
    # M = np.c_[np.ones((len(x), 1)), x, y, x ** 2, x * y, y ** 2,\
    #              x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
    #              x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
    #              x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]


    # Calculate beta
    A = np.arange(1, degree + 2)
    rows = np.sum(A)
    beta = (np.linalg.inv(M.T.dot(M) + l * np.identity(rows))).dot(M.T).dot(z)

    return beta



