import numpy as np

def ols(x, y, z):
    #x: vector of size(n, 1)
    #y: vector of size(n,1)
    # z: vector of size(n,1)

    xyb = np.c_[np.ones_like(x), x, y, x*x, y*y, x*y]
    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)

    return beta
