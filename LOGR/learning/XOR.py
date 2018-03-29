import numpy as np
import matplotlib.pyplot as plt
from  ..functions.activation_functions import sigmoid
# from  .cross_entropy_loss import cross_entropy
from ..functions.loss_functions import cross_entropy
from ..functions.regularisation_functions import l2_regularization 

'''
    *** CREATING XOR DATA ***
'''

# There are only 4 points in XOR problem. 
# Only one point can be true to resolve in True value

N = 4
D = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

T = np.array([0, 1, 1, 0])

plt.scatter(X[:,0],X[:,1],c=T)
plt.show()

# Can see that the values can't be cut in any direction to get 100%

'''
    *** START SOLVING ***
'''

# create column of bias values just of ones
b = np.array([[1]*N]).T

# Add another dimension to input to turn it into a 3D problem
# done by multiplying the x and y to a new variable to
# make data linearly separable
xy = np.matrix(X[:,0]*X[:,1].T).T

print(X.shape, b.shape, xy.shape)

# concatenate to input data
Xb = np.array(np.concatenate((b,xy,X), axis=1))

# randomly initialise weights
w = np.random.rand(D+2) # the +2 accounts for the ones biases and the new xy matrix

z = Xb.dot(w)

# Cause the classes to be either 0 or 1. 
Y = sigmoid(z)

lr = 0.001 # predetermined
error = []
# number of iterations also predetermined to be 5000
for i in range(8000):
    # the cross_entropy_loss function works fine from the loss function folder
    # however, it returns an error of being divided by 0? so can't visualise loss changes
    e = cross_entropy(T,Y)
    error.append(e)
    # print error every 100 steps to visualise change in error
    if i % 100 == 0:
        print(e)
    
    # since weights are being updated upstream, 
    # you're technically using the derivatives of the regularisation
    w += lr *( np.dot((T-Y).T,Xb) - l2_regularization(0.01,w))
    Y = sigmoid((Xb.dot(w)))

plt.plot(error)
plt.title("Cross Entropy")

print('Final w:', w)
# when classifying you're rounding. Any thing that is different is counted as wrong
print('Final classification rate:', 1- np.abs(T-np.round(Y)).sum()/N)

# Then draw a plan through the data