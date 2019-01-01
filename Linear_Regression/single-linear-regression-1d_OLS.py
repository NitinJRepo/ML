#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:43:59 2018

@author: nitin
"""
# shows how linear regression analysis can be applied to 1-dimensional data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

housing = pd.read_csv('./data/RestaurentData.txt')

# Normalisisng the data
housing = (housing - housing.mean())/housing.std()
housing.head()

'''
# Assign feature variable X
X = housing['area']
# Assign response variable to y
Y = housing['price']
'''
X = housing.iloc[:, 0].values
Y = housing.iloc[:, 1].values


# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

# let's plot the data to see what it looks like
plt.scatter(X, Y)
plt.show()


# apply the equations we learned to calculate a and b

# denominator is common
# note: this could be more efficient if
#       we only computed the sums and means once
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# let's calculate the predicted Y
Yhat = a*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

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
print("R-squared = ", r2)


