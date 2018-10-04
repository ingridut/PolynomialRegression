# -*- coding: utf-8 -*-
""" 
Created on Mon Sep 17 12:59:58 2018
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from fit_and_test import fit_and_test
from plot_terrain import plot_terrain
from k_fold import k_fold_validation
from RidgeRegression import RidgeRegression
from MSE import MeanSquaredError
from Lasso import Lasso

#Read data
terrain_over_Norway = imread('SRTM_data_Norway_1.tif')

#Choose to lo look at a little part of the data-set:
terrain = terrain_over_Norway[0:100,0:100]

"""
#plot of the original
plt.figure()
plt.title('Terrain')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
#plt.savefig('oiginal_subset.png')
plt.show()
"""

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

#np.random.shuffle(X)                #shuffle rows in x


x = X[:,0, np.newaxis] / (num_cols)
y = X[:,1, np.newaxis] / (num_cols)
z = X[:,2, np.newaxis] 


#Try with the whole set for training and testing
#fit_and_test(x, y, z, x, y, z)

#find the optimal value for lamda in ridge regression
test_error = np.zeros((100))
x_train = x[0:49,0, np.newaxis]
y_train = x[0:49,0, np.newaxis]
z_train = x[0:49,0, np.newaxis]
x_test = x[50:100,0, np.newaxis]
y_test = x[50:100,0, np.newaxis]
z_test = x[50:100,0, np.newaxis]


x_values = np.linspace(0.1,1,100)
index = 0
for Lambda in x_values:
    beta = RidgeRegression(x_train, y_train, z_train, l=Lambda)
    MSE = MeanSquaredError(x_test, y_test, z_test, beta)
    test_error[index] = MSE
    index += 1

plt.figure()
#plt.title('MSE for different lambda values')
plt.plot(x_values,test_error)
plt.xlabel('lambda values')
plt.ylabel('Mean squered errer')
plt.savefig('lambda.png')
plt.show()


index = 0
x_values = np.linspace(0.00000001,0.1,100)
for alfa in x_values:
    beta = Lasso(x_train, y_train, z_train, a = alfa)
    MSE = MeanSquaredError(x_test, y_test, z_test, beta.reshape(21,1))
    test_error[index] = MSE
    index += 1
 
#plt.title('MSE for different alpha values')
plt.plot(x_values,test_error)
plt.xlabel('alpha values')
plt.ylabel('Mean squered errer')
plt.savefig('alpha.png')
plt.show()    


#try with k-fold validation and find the best beta-value
result_OLS = k_fold_validation(x, y, z, 'OLS')
result_Ridge = k_fold_validation(x, y, z, 'Ridge')
result_Lasso = k_fold_validation(x, y, z, 'Lasso')

betas_OLS = result_OLS[0];
mse_OLS = result_OLS[1]
r2_OLS = result_OLS[2]
min_mse_OLS = 0
best_R2_OLS = 0

betas_Ridge = result_Ridge[0];
mse_Ridge = result_Ridge[1]
r2_Ridge = result_Ridge[2]
min_mse_Ridge = 0
best_R2_Ridge = 0

betas_Lasso = result_Lasso[0];
mse_Lasso = result_Lasso[1]
r2_Lasso = result_Lasso[2]
min_mse_Lasso = 0
best_R2_Lasso = 0

for i in range(0, len(result_OLS[0])):
    if r2_OLS[i] > best_R2_OLS:
        best_beta_OLS = betas_OLS[i]
        min_mse_OLS = mse_OLS[i]
        best_R2_OLS = r2_OLS[i]
    if r2_Ridge[i] > best_R2_Ridge:
        best_beta_Ridge = betas_Ridge[i]
        min_mse_Ridge = mse_Ridge[i]
        best_R2_Ridge = r2_Ridge[i]
    if r2_Lasso[i] > best_R2_Lasso:
        best_beta_Lasso = betas_Lasso[i]
        min_mse_Lasso = mse_Lasso[i]
        best_R2_Lasso = r2_Lasso[i]
  
print('Best fit OLS')      
plot_terrain(100, 100, best_beta_OLS)
print('Mean squared error {} '.format(min_mse_OLS))
print('R2 score {} '.format(best_R2_OLS))

print('Best fit Ridge')      
plot_terrain(100, 100, best_beta_Ridge)
print('Mean squared error {} '.format(min_mse_Ridge))
print('R2 score {} '.format(best_R2_Ridge))

print('Best fit Lasso')      
plot_terrain(100, 100, best_beta_Lasso)
print('Mean squared error {} '.format(min_mse_Lasso))
print('R2 score {} '.format(best_R2_Lasso))
