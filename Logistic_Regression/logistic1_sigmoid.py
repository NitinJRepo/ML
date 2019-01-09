#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:41:34 2018

@author: nitin
"""

import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)

""" 
Creating and Adding a bias term:
    
Usually we just add a column of ones to the original data (X) and 
include the bias term in the weights (W)  

"""
ones = np.ones((N,1))   

# add a column of ones to the original data (X)
Xb = np.concatenate((ones, X), axis=1)

# Randomly initialize weights
# include the bias term in the weights (W)
W = np.random.randn(D + 1)

#===============================
# calculate the model output
#===============================
# Calculate the dot product between each row of X and W
z = Xb.dot(W)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print ("predictions :", sigmoid(z))



    