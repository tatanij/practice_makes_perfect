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
    k = n/4
    j = n/2
    i = n *0.75
    X[:k] = np.random.random((k,d))/2 + 0.5
    X[k:j] = np.random.random((k,d))/2
    X[j:i] = np.random.random((k,d))/2 + np.array([[0,0.5]])
    X[i:] = np.random.random((k,d))/2 + np.array([[0.5,0]])
    Y = np.array([0]*100 + [1]*100)
    return X,Y

#def generate_donut(N=200,R_inner=5, R_outer=10):
    
    
    