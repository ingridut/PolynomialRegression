import numpy as np

def MeanSquaredError(x, y, z, beta):
    """
    Calculates the Mean Squared Error
    :param y: numpy vector with y data, size (n, 1)
    :param x: numpy vector with x data, size (n, 1)
    :param beta: model
    :return: Mean squared error
    """
    # Calculate z_hat, the predicted z-values
    M = np.c_[np.ones((100 ** 2, 1)), x, y, x ** 2, x * y, y ** 2,\
                 x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
                 x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
                 x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]
    z_hat = M.dot(beta)

    # Calculate MSE
    MSE = 0
    for i in range(0, len(z)):
        MSE += np.power(z[i, 0] - z_hat[i, 0], 2)
    MSE = MSE/len(z)

    return MSE
