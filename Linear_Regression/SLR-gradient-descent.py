"""
Created on Sun Nov 25 13:03:44 2018

@author: nitin
"""
# shows how linear regression analysis can be applied to 1-dimensional data (using gradient descent)
# NOT READY
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the data
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
#X = housing.iloc[:, :-1].values
X = housing.iloc[:, 0].values
Y = housing.iloc[:, 1].values

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)

# let's plot the data to see what it looks like
plt.scatter(X, Y)
plt.show()

N = X.shape[0]

m_current=0
c_current=0
iters=1000
learning_rate=0.01
costs = [] # keep track of squared error cost

for i in range(iters):
    Yhat = (m_current * X) + c_current
    delta = Y - Yhat
    
    #mse = sum([data**2 for data in delta]) / N
    mse = delta.dot(delta) / N
    
    m_gradient = -(2/N) * sum(X * delta)
    c_gradient = -(2/N) * sum(delta)
    m_current = m_current - (learning_rate * m_gradient)
    c_current = c_current - (learning_rate * c_gradient)
    
    #print("cost = ",mse)
    costs.append(mse)

plt.plot(costs)
plt.show()

# let's plot everything together to make sure it worked
plt.scatter(X, Y) # Plotting scatter points
plt.plot(X, Yhat,'r') # Plotting the line
plt.show()


# R-Squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)

#================================
'''# Append bias term to X
N = X.shape[0]

# Add a column of ones to the original data (X)
ones = np.ones((N,1))

#Xb = np.concatenate((ones, X), axis=1)
Xb = np.hstack((ones, X))

D = Xb.shape[1]

# let's try gradient descent
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
w = w[:, np.newaxis]
learning_rate = 0.001

for t in range(1000):
  # update w
  Yhat = Xb.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*Xb.T.dot(delta)

  # find and store the cost
  mse = delta.T.dot(delta) / N
  costs.append(mse)


# plot the costs
plt.plot(costs)
plt.show()
'''
#================================

