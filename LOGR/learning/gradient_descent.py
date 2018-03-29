import  numpy as np
from .logr_ex import sigmoid
from .cross_entropy_loss import cross_entropy, cross_entropy_loss
from .l2_regularization import l2_reg

# creating random 2-dimensional data
N = 100
D = 2 

X = np.random.randn(N,D)

X[:50, :] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:, :] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

#randomly initialise weights
w = np.random.randn(D+1)

# calculate model output
z = Xb.dot(w)

Y = sigmoid(z)

# implementation of gradient descent 
lr = 0.1
# do 100 epochs
for i in range(100):
    # print loss every 10 epochs
    if i % 10 == 0:
        print(cross_entropy_loss(T,Y,N))

    #update weights
    # w += lr * np.dot((T-Y).T,Xb)

    # update weights w regularisation
    w += lr * np.dot((T-Y).T, Xb) - l2_reg(0.1, w, True)
    # recalculate output
    Y = sigmoid(Xb.dot(w))

print('Final w:',w)