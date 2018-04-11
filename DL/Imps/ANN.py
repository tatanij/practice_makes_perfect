import numpy as np 
import sys
sys.path.append('../../functions/')
from activation_functions import softmax, sigmoid, relu

class ANN(object):
    def __init__(self, X, Y, n_in, n_hidden, n_out, learning_rate=0.001, lambda=None, epochs=100):
        # initialise weights
        self.w0 = np.random.randn(n_in,n_hidden)
        self.b0 = np.random.randn(n_hidden)


    def forward(X,Y): 
        

    def backward(X, Y, self.Yhat):

    
    def classfication(Y,self.Yhat):