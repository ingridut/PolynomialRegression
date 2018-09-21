import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from frankeFunction import FrankeFunction
from scipy import misc

def LassoRegression(x, y, z, degree=5, l=0.1):
    # Arranging data into 2x50 matrix
    a = np.array(x) #inputs
    b = np.array(y) #outputs

    #Split into training and test
    X_train = a[:15, np.newaxis]        # picks 15 first elements of x
    X_test = a[15:, np.newaxis]         # picks 5 (remaining) last elements of x
    Y_train = b[:15]                    # same, but for y axis
    Y_test = b[15:]
    ##########


    print ("X_train: ", X_train.shape)
    print ("y_train: ", Y_train.shape)
    print ("X_test: ", X_test.shape)
    print ("y_test: ", Y_test.shape)

    lasso = PolynomialFeatures(degree)
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train, Y_train)             # lasso.fit([x,y],z)
    predl = lasso.predict(X_test)
    print ("Lasso Coefficient: ", lasso.coef_)
    print ("Lasso Intercept: ", lasso.intercept_)
    plt.scatter(X_test, Y_test,color='green', label="Training Data")
    plt.plot(X_test, predl, color='blue', label="Lasso degree: %d" % degree)
    plt.legend()
    plt.show()
    return 5

##########
#terrain1 = misc.imread('data.tif',flatten=0)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
#x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y)
beta = LassoRegression(x,y,z)
