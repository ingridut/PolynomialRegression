# -*- coding: utf-8 -*-
""" 
Created on Mon Sep 17 12:59:58 2018
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from fit_and_test import fit_and_test
from plot_terrain import plot_terrain
from Analysis import k_fold_validation
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm


#Read data
terrain_over_Norway = imread('SRTM_data_Norway_1.tif')

#Choose to lo look at a little part of the data-set:
terrain = terrain_over_Norway[0:100,0:100]

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

x = X[:,0, np.newaxis] / (num_cols)
y = X[:,1, np.newaxis] / (num_cols)
z = X[:,2, np.newaxis] 

#Try with the whole set for training and testing
fit_and_test(x, y, z, x, y, z)

#try with k-fold validation and find the best beta-value
result = k_fold_validation(x,y,z)
betas = result[0];
mse = result[1]
r2 = result[2]

#best_beta = np.array((21,1))
min_mse = 0
best_R2 = 0
for i in range(0, len(result[0])):
    if r2[i] > best_R2:
        best_beta = betas[i]
        min_mse = mse[i]
        best_R2 = r2[i]
        
plot_terrain(100, 100, best_beta)
print(min_mse)
print(best_R2)