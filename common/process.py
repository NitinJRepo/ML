#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:40:47 2018

@author: nitin
"""
import numpy as np
import pandas as pd
import os

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath('__file__')))

def get_data():
    df = pd.read_csv(dir_path + "/ecommerce_data.csv")
    #df = pd.read_csv(dir_path + "/../common/ecommerce_data.csv")
    #df.head()
    
    # easier to work with numpy array
    data = df.values
    
    # Shuffle it
    np.random.shuffle(data)
    
    # Split features and labels 
    X = data[:, :-1]    # X is going to be everything upto the last column
    Y = data[:, -1].astype(np.int32)     # Y will be the last column 

    #=============================================================
    # One hot encoding of categorical data
    #=============================================================
    # 1. Create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] # Non-categorical data
    
    # 2. One hot encode
    for row in range(N):
        t = int(X[row, D-1])       # The int() method returns an integer object from any number or string.
        X2[row, t+D-1] = 1
    
    # 3. Assign X2 back to X, since we don't need original anymore
    X = X2

    #==========================================
    # Normalization of the numerical columns
    #==========================================
    # 1. split train and test
    Xtrain = X[:-100]            # everything except last 100 elements
    Ytrain = Y[:-100]            # everything except last 100 elements
    Xtest = X[-100:]             # last 100 elements
    Ytest = Y[-100:]             # last 100 elements
    
    # 2. normalize columns 1 and 2
    for i in (1, 2):
        m = Xtrain[:,i].mean()
        s = Xtrain[:,i].std()
        Xtrain[:,i] = (Xtrain[:,i] - m) / s
        Xtest[:,i] = (Xtest[:,i] - m) / s
    
    return Xtrain, Ytrain, Xtest, Ytest


# For logistic class we only want binary data, so we don't want full dataset
def get_binary_data():
    # return only the data from the first 2 classes
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain <= 1]        # All Xtrain elements where Ytrain <=1
    Y2train = Ytrain[Ytrain <= 1]        # All Ytrain elements where Ytrain <=1
    X2test = Xtest[Ytest <= 1]
    Y2test = Ytest[Ytest <= 1]
    
    return X2train, Y2train, X2test, Y2test

 