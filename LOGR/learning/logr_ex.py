import numpy as np
import matplotlib.pyplot as plt 

N = 100
D = 2 #2 variables affecting the output

X = np.random.rand(N,D) #Creating random data 
w0 = np.array([1]*N).T #need it to be 2 dimensional to have n rows

X = np.insert(X, 0, w0, axis=1)

# intialise weights randomly 
w = np.random.randn(D+1)

#do dot product to get input for sigmoid. 
#remember that the sigmoid function will return 0 or 1 to classify an input
# print(X.shape, w.shape)
z = X.dot(w) #i.e. wTx 

def sigmoid(z):
    return 1/(1+np.exp(-z))

# print(sigmoid(z))