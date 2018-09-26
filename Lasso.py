from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

def Lasso(x, y, z, degree):

    X = np.c_[x, y]
    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(X)

    clf = linear_model.LassoCV()
    clf.fit(X_, z)
    beta = clf.coef_

    return beta