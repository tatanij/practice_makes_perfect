import numpy as np 
from ..functions.loss_functions import cross_entropy

Y_hat = np.array([
       [ 0.01364086,  0.83614423,  0.52476459,  0.57620765],
       [ 0.42054362,  0.09608707,  0.98375077,  0.81954481],
       [ 0.51997766,  0.55742316,  0.39389826,  0.89670172]
        ])
Y_actual = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1]
        ])

calced_cse = np.array([])

print('Testing forward pass:',cross_entropy(Y_actual,Y_hat))
