# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:28:13 2018

@author: Betina
"""
from OrdinaryLeastSquare import ols
from RidgeRegression import RidgeRegression
from Lasso import Lasso
from MSE import MeanSquaredError
from plot_terrain import plot_terrain
from Analysis import R2
from predict import predict
import numpy as np

def fit_and_test(x_train, y_train, z_train, x_test, y_test, z_test):
    
    num_cols = 100
    num_rows = 100
    
    #fit data with Ordinary Least Squares
    print('OLS')
    beta_OLS = ols(x_train, y_train, z_train)
    MSE_OLS = MeanSquaredError(x_test, y_test, z_test, beta_OLS)
    R2_OLS = R2(z_test, predict(x_test, y_test, beta_OLS))
    plot_terrain(num_rows, num_cols, beta_OLS)
    print(MSE_OLS)
    print(R2_OLS)
    
    #fit data with Ridge Regression, test model with testset and calculate MSE
    print('Ridge')
    beta_Ridge = RidgeRegression(x_train, y_train, z_train, l = 10)
    MSE_Ridge = MeanSquaredError(x_test, y_test, z_test, beta_Ridge)
    R2_Ridge = R2(z_test, predict(x_test, y_test, beta_Ridge))
    plot_terrain(num_rows, num_cols, beta_Ridge)
    print(MSE_Ridge)
    print(R2_Ridge)
    print(beta_Ridge)
    
    #fit data with Lasso Regression, test model with testset and calculate MSE
    print('Lasso')
    #beta_Lasso = np.array((21,1))
    beta_Lasso = Lasso(np.array(x_train), np.array(y_train), np.array(z_train), 5).reshape((21, 1))
    MSE_Lasso = MeanSquaredError(x_test, y_test, z_test, beta_Lasso)
    R2_Lasso = R2(z_test, predict(x_test, y_test, beta_Lasso))
    print(MSE_Lasso)
    print(R2_Lasso)