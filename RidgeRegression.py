import numpy as np

def RidgeRegression(x, y, z, degree=5, l=0.1):
    """
    Linear regression using Ridge method
    :param x: numpy vector of size (n, 1)
    :param y: numpy vector of size (n, 1)
    :param degree: degree of polynomial fit
    :param l: Ridge
    :return: numpy array, size (rows, 1)

    """
    """
    A = np.arange(1, degree+2)
    rows = np.sum(A)
    M = np.empty((len(x), rows))
    row = 0
    for i in range(degree+1):
        for j in range(degree+1):
            if i+j <= 5:
                M[:, row] = np.power(x[:, 0], i)*(np.power(y[:, 0], j))
                row += 1
    """

    # Calculate matrix with powers of x and y
    M = np.c_[np.ones((100, 1)), x, y, x ** 2, x * y, y ** 2,\
                 x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
                 x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
                 x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]

    # Calculate beta
    A = np.arange(1, degree + 2)
    rows = np.sum(A)
    beta = (np.linalg.inv(M.T.dot(M) + l * np.identity(rows))).dot(M.T).dot(z)

    return beta



