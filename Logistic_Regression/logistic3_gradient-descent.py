#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# demonstrates how to do gradient descent with numpy matrices.
"""
Created on Tue Oct  2 19:14:36 2018

@author: nitin
"""
import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

N_per_class = N // 2


X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:N_per_class,:] = X[:N_per_class,:] - 2*np.ones((N_per_class,D))

# center the last 50 points at (2, 2)
X[N_per_class:,:] = X[N_per_class:,:] + 2*np.ones((N_per_class,D))

# labels: first N_per_class are 0, last N_per_class are 1
T = np.array([0]*N_per_class + [1]*N_per_class)

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
# 2. Model output (predictions)
# output of logistic regression is  P(Y=1 | X) : "probability of Y equals 1 for given X". The shortcut is only "Y" 

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def calculate_model_output(Xb, W):
    Y =  sigmoid(Xb.dot(W))
    return Y

#=============================================================
# Calculate the cross-entropy error
#  
# The purpose of the cost/error is only to find the weights, because we take the derivative of the cost(in gradient descent) wrt weights.
# The actual value of the cost has no meaning.
#=============================================================

def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

#=================================================================
# Partial derivative of cost function with respect to weights
#=================================================================
def derivative_w(Xb, Y, T):
    return Xb.T.dot(T - Y)  # <-- This is partial derivative of cross-entropy error function
                            # target - prediction
    
#==================================
# Perform Gradient-Descent
#==================================
learning_rate = 0.1
no_of_iteration = 200
costs = []

for i in range(no_of_iteration):
    Y = calculate_model_output(Xb, W)
    
    if i % 10 == 0:
        c = cross_entropy(T, Y)
        print("Error (cost) : ", c)
        costs.append(c)
    
    #*********************************************************************************************************************
    # Update the weights using Gradient-Descent
    # 
    # In this example, we are adding to W because we are maximizing the log likelihood (as opposed to minimizing the negative log likelihood)
    # Maximizing the log likelihood is equal to minimizing cross-entropy error (OR minimize negative log-likelihood)
    # Please note that, following will be true:
    #
    # W -= learning_rate * Xb.T.dot(Y - T)
    # W += learning_rate * Xb.T.dot(T - Y)
    # 
    # One has targets - predictions, the other has predictions - targets.
    #
    # One is minimizing the negative log-likelihood (i.e. cross entropy) (gradient descent) and the other is maximizing the 
    # log-likelihood (gradient ascent).
    #
    # In probability, we have a technique called "maximum likelihood", not "minimum negative likelihood".
    # In machine learning, we talk about "minimizing cost", not "maximizing negative cost".
    # But if the cost is the negative likelihood, then you can see why one could equivalently prefer the maximization view 
    # or the minimization view.	
    #*********************************************************************************************************************
    W += learning_rate * derivative_w(Xb, Y, T)
  

print("Final Weights:", W)

#==================================

# plot the data and separating line
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -(W[0] + x_axis*W[1]) / W[2]
plt.plot(x_axis, y_axis)
plt.show()


# plot the cost
plt.plot(costs)
plt.show()