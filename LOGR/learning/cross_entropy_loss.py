import numpy as np

def cross_entropy_loss(T,Y,N):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E 

def cross_entropy(T, pY, dy=False):
    if dy:
        return -np.mean(T*(1/pY))
    else:
        return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))

