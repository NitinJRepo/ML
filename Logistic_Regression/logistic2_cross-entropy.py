#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:39:53 2018

@author: nitin
"""

import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

#======================================================================
# Creating and Adding a bias term:
    
# Usually we just add a column of ones to the original data (X) and 
# include the bias term in the weights (W)
#======================================================================  
ones = np.ones((N,1))   

# 1. add a column of ones to the original data (X)
Xb = np.concatenate((ones, X), axis=1)

# Randomly initialize weights
# 2. include the bias term in the weights (W)
W = np.random.randn(D + 1)

#===============================
# calculate the model output
#===============================
# 1. Calculate the dot product between each row of X and W
z = Xb.dot(W)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 2. Model output (predictions)
Y = sigmoid(z)

# calculate the cross-entropy error
#===============================================================================
# Q. Should the cross entropy error not be multiplied by 1/N ?
# A: Not necessary, because by the update rule: W = W - learning_rate * G, 
#    any constant is absorbed by the learning rate.
#===============================================================================
# cross entropy is : -target * log(prediction probability)
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print ("Error : " , cross_entropy(T, Y))

#==================================================================
# try it with our closed-form solution
w = np.array([0, 4, 4])

# calculate the model output
z = Xb.dot(w)  
Y = sigmoid(z)

# calculate the cross-entropy error
print("Error : ", cross_entropy(T, Y))
#=================================================================
