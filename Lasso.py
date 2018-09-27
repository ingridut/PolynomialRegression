from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

def Lasso(x, y, z, degree=5, a=0.0000001):

    X = np.c_[x, y]
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(X)

    clf = linear_model.Lasso(alpha=a, max_iter=10000, tol=0.00001, fit_intercept=False)
    clf.fit(X_, z)
    beta = clf.coef_

    return beta