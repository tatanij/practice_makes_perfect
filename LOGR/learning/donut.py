
import numpy as np
import matplotlib.pyplot as plt
from  ..functions.activation_functions import sigmoid
from  .cross_entropy_loss import cross_entropy
from ..functions.regularisation_functions import l2_regularization 

# using matplotlib for visualisation of the problem.

N = 1000
D = 2

# 2 radiuses, 1 inner and 1 outer
R_inner = 5
R_outer = 10

q = int(N/2)

# set uniformly distributed variable 
# for half the data that depends on the inner radius
R1 = np.random.randn(q) + R_inner

# generate angles - polar coordinates which are uniformly distributed
theta = 2*np.pi*np.random.random(q)

# convert polar coordinates into X,Y (a.k.a T) coordinates and transpose so N goes along rows
X_inner = np.concatenate([[R1*np.cos(theta)],[R1 * np.sin(theta)]]).T

# do same thing for outer radius
R2 =  np.random.randn(q) + R_outer
theta2 = 2*np.pi*np.random.random(q)
X_outer =np.concatenate([[R2*np.cos(theta2)],[R2 * np.sin(theta2)]]).T

X = np.concatenate([X_inner,X_outer])
T = np.array([0]*(q)+[1]*(q))

# visualise what it looks like so far
plt.scatter(X[:,0],X[:,1], c=T)
plt.show()

'''
 ***    START SOLVING   ***
'''

# As seen logistic may not be good for this problem since there
# is no line that can separate the classes. However it is possible.

# create a columns of 1s for bias term.
ones = np.array([[1]*N]).T

# create a column which represents the radius of a point
# this helps make the data points linearly separable
r = np.array([np.sqrt(X[i,:].dot(X[i,:])) for i in range(N)]).reshape(N,1) # might need to be changed

# concatenate X, ones and radii together
Xb = np.concatenate((ones, r, X), axis=1)

# randomly initialise weights
w = np.random.rand(D+2) # the +2 accounts for the ones biases and the radi

z = Xb.dot(w)

# Cause the classes to be either 0 or 1. 
Y = sigmoid(z)

lr = 0.0001 # predetermined
error = []
# number of iterations also predetermined to be 5000
for i in range(5000):
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