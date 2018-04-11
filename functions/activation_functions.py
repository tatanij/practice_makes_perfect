import numpy as np 

def sigmoid(z, w_n=None, diff=None, X=None, dy=False):
    if dy:
        dz = z*(1-z)
        dj = np.sum(diff.dot(w_n))
        return X.T.dot(dj * dz)
    else:
        return 1/(1+np.exp(-z))

def tanh(z, dy=False):
    if dy:
        return 1 - pow(tanh(z,False),2)
    else:
        return (np.exp(2*z)-1)/(np.exp(2*z)+1)

def relu(z, dy=False):
    if dy.any():    
        return 1 if z>0 else 0
    else: 
        return np.max(0,z)       

def softplus(z, dy=False):
    if dy:
        return sigmoid(z)
    else:
        return np.log(1+np.exp(z))
    
def softmax(a, Y=None, Z=None, dy=False):
    if not dy:
        return np.exp(a)/np.exp(a).sum(axis=1, keepdims=True)
    else: 
        return np.sum(Z.T.dot((a-Y)),axis=0),a # also returning the result of the sigmoid function
        