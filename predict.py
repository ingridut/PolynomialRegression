# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:40:11 2018
Makes a prediction given a vector x, a vector y and a model beta
@author: Betina
"""
import numpy as np
def predict(x, y, beta):
    """
    Calculates the Mean Squared Error
    :param y: numpy vector with y data, size (n, 1)
    :param x: numpy vector with x data, size (n, 1)
    :param beta: model
    :return: Mean squared error
    """
    # Calculate z_hat, the predicted z-values
    M = np.c_[np.ones((len(x), 1)), x, y, x ** 2, x * y, y ** 2,\
                 x ** 3, x ** 2 * y, x * y ** 2, y ** 3,\
                 x ** 4, x ** 3 * y, x ** 2 * y ** 2, x * y ** 3, y ** 4,\
                 x ** 5, x ** 4 * y, x ** 3 * y ** 2, x ** 2 * y ** 3, x * y ** 4, y ** 5]
    return M.dot(beta)