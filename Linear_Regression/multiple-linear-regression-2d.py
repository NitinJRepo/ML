#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:59:40 2018

@author: nitin
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# load the data
X = []
Y = []
for line in open('./data/data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) # add the bias term
    Y.append(float(y))

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)


# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


#========================================
# Normal Equation (Closed form soultion)
#========================================
# apply the equations we learned to calculate a and b
# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)


# Metrics for linear regression
#=================================
d1 = Y - Yhat
d2 = Y - Y.mean()

# RMSE
N = X.shape[0]
mse = (d1).dot(d1) / N
print("mse = ", mse)
print("RMSE = ", np.sqrt(mse))

# R-Squared: determine how good the model is by computing the r-squared
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)
