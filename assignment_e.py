# -*- coding: utf-8 -*-
""" 
Created on Mon Sep 17 12:59:58 2018
"""
import numpy as np
from imageio import imread
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from RidgeRegression import RidgeRegression
#from LassoRegression import LassoRegression
from MSE import MeanSquaredError

#percent of datat used as testset
p = 0.75

#Read datat
terrain = imread('SRTM_data_Norway_1.tif')
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

#shuffle rows in x and devide into trainingset an testset
np.random.shuffle(X)
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
#beta_OLS = OLS(x_train, y_train, z_train)
#MSE_OLS = MeanSquaredError(x_test, y_test, z_test, beta_OLS)

#fit data with Ridge Regression, test model with testset and calculate MSE
beta_Ridge = RidgeRegression(x_train, y_train, z_train)
MSE_Ridge = MeanSquaredError(x_test, y_test, z_test, beta_Ridge)


#fit data with Lasso Regression, test model with testset and calculate MSE
#beta_Lasso= LassoRegression(x_train, y_train, z_train)
#MSE_Lasso = MeanquaredError(x_test, y_test, z_test, beta_Lasso)

print(MSE_Ridge)