import numpy as np 

def sigmoid(z, dy=False):
    if dy:
        return z
    else:
        return 1/(1+np.exp(-z))

def tanh(z, dy=False):
    if dy:
        return 1 - pow(tanh(z,False),2)
    else:
        return (np.exp(2*z)-1)/(np.exp(2*z)+1)

def relu(z, dy=False):
    if dy:
        return 1 if z>0 else 0
    else: 
        return np.max(0,z)       

def softplus(z, dy=False):
    if dy:
        return sigmoid(z)
    else:
        return np.log(1+np.exp(z))