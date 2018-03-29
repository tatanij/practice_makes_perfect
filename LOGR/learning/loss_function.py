import numpy as np 
import matplotlib.pyplot as plt 
from logr_ex import sigmoid


N = 100
D = 5


# random values initiated
X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D)) # center values between -2
X[50:,:] = X[50:,:] + 2*np.ones((50,D)) # and +2

# targets definition
T = np.array([0]*50 +[1]*50)

ones = np.array([[1]*N]).T #bias initiated with ones. i.e. b0
Xb = np.concatenate((ones,X), axis=1)

# initialise weights randomly
w = np.random.randn(D+1)

# calculate the model about 
z = Xb.dot(w)

# def sigmoid(z):
#     return 1/(1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T,Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(cross_entropy(T,Y))

# closed form solution to logistic regress. 
# works because there is equal variane in both classes
# weights therefore only depend on the mean

w = np.array([0,4,4])
print(w.shape)
z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T,Y))

'''
def softmax(signal, dy = False):
    # Calculate activation signal
    e_x = np.exp(signal - np.max(signal, axis=1, keepdims=True))
    signal  = e_x/np.sum(e_x,axis=1, keepdims=True)

    if dy:
        return np.ones(signal.shape)
    else:
        # return the activation signal
        return signal
        '''