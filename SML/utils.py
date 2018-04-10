import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def get_mnist_data(fn, limit=None):
    data = pd.read_csv(fn).as_matrix()
    #shuffle data 
    np.random.shuffle(data)
    
    X = data[:,1:]/255.0 # normalise the data since they are greyscale image values
    Y = data[:,0] # y value is given at the start
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X,Y

def generate_xor(n=200, d=2):
    # generate a random dataset of n samples
    X = np.zeros((n,d))
    k = int(n/4)
    j = int(n/2)
    i = int(n *0.75)
    
    # treating anything above 0.5 and below 1 as true and anything below 0.5 and between 0 as false
    X[:k] = np.random.random((k,d))/2 + 0.5
    X[k:j] = np.random.random((k,d))/2
    X[j:i] = np.random.random((k,d))/2 + np.array([[0,0.5]])
    X[i:] = np.random.random((k,d))/2 + np.array([[0.5,0]])
    Y = np.array([0]*j + [1]*j) # create cloud of points from 0 to 1 on X-axis
    return X,Y

def generate_simple_xor():
    X = np.array([[0,1],[1,0]])
    Y = np.array([0,1,1,0])
    return X,Y

def generate_donut(N=200,R_inner=5, R_outer=10):
        
    q = int(N/2)

    # create set for inner and outer radius
    R1 = np.random.randn(q) + R_inner
    R2 = np.random.randn(q) + R_outer

    # generate uniformly distributed polar coordinates - an angle which is distributed along (0,2*pi)
    theta = 2*np.pi*np.random.random(q)  # 2 * pi * r - but for a set of q random numbers
    theta1 = 2*np.pi*np.random.random(q) 

    # convert polar coordinates for X,Y coordinates and transopose so N goes along the rows
    X_inner = np.concatenate([[R1*np.cos(theta)],[R1*np.sin(theta)]]).T
    X_outer = np.concatenate([[R2*np.cos(theta1)],[R2*np.sin(theta1)]]).T

    X = np.concatenate([X_inner,X_outer])
    Y = np.array([0]*(q) + [1] *(q))

    return X,Y
        
    