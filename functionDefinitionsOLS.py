from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import randrange, uniform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%matplotlib qt

#FrankeFunction - for simulation of data
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2)- 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(x.shape[0], x.shape[1])

#Ordinary Least Squared function
#def ols(x, y, z):
#    #x: vector of size(n, 1)
#    #y: vector of size(n,1)
#    # z: vector of size(n,1)
#
#    xyb = np.c_[np.ones_like(x), x, y, x*x, y*y, x*y]
#    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
#    return beta
    
#Ordinary Least Squared function
def ols(x, y, z, degree = 2):
    #x: vector of size(n, 1)
    #y: vector of size(n,1)
    # z: vector of size(n,1)
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform(xyb_)
    beta = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
    return beta

def makeTestValues(x,y, degree=2):
    xyb_ = np.c_[x, y]
    poly = PolynomialFeatures(degree)
    xyb = poly.fit_transform([x,y])
    return xyb

#Mean Squared Error
def mse(zReal, zPredicted):
    mse = np.mean((zReal-zPredicted)**2)
    return mse           

def bias(zReal, zPredicted):
    bias = np.mean( (zReal- np.mean(zPredicted))**2 )
    return bias
    
def var(zReal, zPredicted):
    var = np.mean( (zPredicted - np.mean(zPredicted))**2 )
    return var

#Mean value of the function
def Mean(z):
    meanValue = np.mean(z)
    return meanValue

#R2 score function
def R2(zReal, zPredicted):
    meanValue = Mean(zReal)
    numerator = np.sum((zReal - zPredicted)**2)
    denominator = np.sum((zReal - meanValue)**2)
    result = 1 - (numerator/denominator)
    return result