import numpy as np 

def mse(Y,Yhat, dy=False):
    return 2*(np.sum(Y-Yhat)) if dy else np.mean(np.sum(np.square((Y-Yhat)))) 

def cross_entropy(Y,Yhat,dy=False):
    if dy:
         return Yhat-Y
    else:
        return -np.mean(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat))
