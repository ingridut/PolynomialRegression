import numpy as np

def MeanSquaredError(y, y_hat):
    MSE = 0
    for i in range(0, len(y)-1):
        MSE += np.power(y(i) - y_hat(i), 2)
    MSE = MSE/len(y)

    return MSE
