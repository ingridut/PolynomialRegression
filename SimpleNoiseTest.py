import numpy as np
from RidgeRegression import RidgeRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from MSE import MeanSquaredError

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

if __name__ == '__main__':
    runTest()