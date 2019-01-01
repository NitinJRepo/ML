#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:54:41 2018

@author: nitin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# load the data
housing = pd.read_csv('./data/Housing.csv')

# Normalisisng the data
housing = (housing - housing.mean())/housing.std()
housing.head()

X = housing.iloc[:, :-1].values
Y = housing.iloc[:, 1].values


# let's plot the data to see what it looks like
plt.scatter(X, Y)
plt.show()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

y_pred = regressor.predict(X)

# Visualising the results
plt.scatter(X, Y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Area Vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

print("R-Square = ", r2_score(Y, y_pred))

