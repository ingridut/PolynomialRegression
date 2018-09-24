# -*- coding: utf-8 -*-
""" 
Created on Mon Sep 17 12:59:58 2018
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from OrdinaryLeastSquare import ols
from RidgeRegression import RidgeRegression
#from LassoRegression import LassoRegression
from MSE import MeanSquaredError
from plot_terrain import plot_terrain

#percent of datat used as testset
p = 0.75

#Read data
terrain = imread('SRTM_data_Norway_1.tif')

#plot of the original
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

num_rows = len(terrain)
num_cols = len(terrain[0])
num_observations = num_rows*num_cols
X = np.zeros((num_observations,3))

#make a matrix with all the values from the data on the form [x y z]
index = 0;
for i in range(0, num_rows):
    for j in range(0, num_cols):
        X[index,0] = i              #x
        X[index,1] = j              #y
        X[index,2] = terrain[i,j]   #z
        index += 1

np.random.shuffle(X)                #shuffle rows in x

#devide into trainingset an testset and devide this into x, y and z-vectors
split = round(p*num_observations)
train = X[0:split,:]
test = X[split+1:,:]
x_train = train[:,0, np.newaxis]
y_train = train[:,1, np.newaxis]
z_train = train[:,2, np.newaxis]
x_test = test[:,0, np.newaxis]
y_test = test[:,1, np.newaxis]
z_test = test[:,2, np.newaxis]

#fit data with Ordinary Least Squares
beta_OLS = ols(x_train, y_train, z_train)
MSE_OLS = MeanSquaredError(x_test, y_test, z_test, beta_OLS)
plot_terrain(num_rows, num_cols, beta_OLS)

#fit data with Ridge Regression, test model with testset and calculate MSE
#beta_Ridge = RidgeRegression(x_train, y_train, z_train)
#MSE_Ridge = MeanSquaredError(x_test, y_test, z_test, beta_Ridge)
#plot_terrain(num_rows, num_cols, beta_Ridge)

#fit data with Lasso Regression, test model with testset and calculate MSE
#beta_Lasso= LassoRegression(x_train, y_train, z_train, x_test)
#MSE_Lasso = MeanquaredError(x_test, y_test, z_test, beta_Lasso)

print(MSE_Ridge)