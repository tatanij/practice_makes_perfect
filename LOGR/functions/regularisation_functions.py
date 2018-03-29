import numpy as np

def l1_regularization(l,w, dy=False):
    if dy: 
        return l*abs(w)
    else: 
        return (l/2)*np.exp(-1*abs(w))

def l2_regularization(l,w,dy=False):
    if dy:
        return l*w
    else:
        return l*np.dot(w.T,w)