import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from frankeFunction import FrankeFunction
from scipy import misc

def LassoRegression(x, y, z, degree=5, l=0.1):
    # Split into training and test
    X_train = x[:15, np.newaxis]        # picks 15 first elements of x
    X_test = x[15:, np.newaxis]         # picks 5 (remaining) last elements of x
    Y_train = y[:15, np.newaxis]        # same, but for y axis
    Y_test = y[15:, np.newaxis]
    Z_train = z[:15, np.newaxis]
    Z_test = z[15:, np.newaxis]

    # Checking dimensions
    print ("X_train: ", X_train.shape)
    print ("Y_train: ", Y_train.shape)
    print ("Z_train: ", Z_train.shape)
    print ("X_test: ", X_test.shape)
    print ("Y_test: ", Y_test.shape)
    print ("Z_test: ", Z_test.shape)

    # Training + testing model and plotting
    # lasso = linear_model.Lasso(alpha=l)
    # #lasso = PolynomialFeatures(degree).fit_transform(X_train)
    # lasso.fit(X_train, Y_train)             # lasso.fit([x,y],z)
    # predl = lasso.predict(X_test)
    # print ("Lasso Coefficient: ", lasso.coef_)
    # print ("Lasso Intercept: ", lasso.intercept_)
    # plt.scatter(X_test, Y_test,color='green', label="Training Data")
    # plt.plot(X_test, predl, color='blue', label="Lasso degree: %d" % degree)
    # plt.legend()
    # plt.show()
    # return 5

    X = np.c_[x,y]                                      # (20, 2)
    # finding design matrix X_
    poly = PolynomialFeatures(degree)
    X_ = .poly.fit_transform(X)                         # (20,) (21 bc deg 5)
    clf = linear_model.LassoCV()
    clf.fit(X_, z)
    print ("X_: ", np.hape(X_))
    #predict = np.append(z,[1])
    #predict_ = predict.reshape(-21,21)
    #beta = clf.predict(predict_)
    beta = clf.coef_                                    # (21,)
    print ("beta val: ", beta)

    x_, y_ = np.meshgrid(x, y)
    x = x_.reshape(-1,1)
    y = y_.reshape(-1,1)
    M = np.c_[x, y]
    M_ = poly.

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_,y_,X_*beta,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    plt.show()

    return 5


if __name__ == '__main__':
    #terrain1 = misc.imread('data.tif',flatten=0)
    x = np.arange(0, 1, 0.05).reshape((20,1))
    y = np.arange(0, 1, 0.05).reshape((20,1))
    z = FrankeFunction(x, y)

    print ("x ", np.shape(x))
    print ("y ", np.shape(y))
    print ("z ", np.shape(z))

    beta = LassoRegression(x,y,z)
