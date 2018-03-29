
# /** 
#   * l1 regularisation define as lasso regression
#   * used to create sparsity in data sets i.e. reduce dimensionality of data
#   * no gaussian prior on w. Therefore need a distribution that can
#   * accept negative absolute value on the exponent - laplace distribution
#   * laplace distributed prior on weights and solving for posterior of w 
#   * with the laplace prior given by p(w) = (l/2)*exp(-l*abs(w))
#   * With the error function, using the same gradient descent alg
#   * it returns 1 if x > -, -1 if x < 0 and 0 if x = 0 
#   * Can combine l1 and l2 regularisation on error functions to create elastinet
# **/

import numpy as np
from logr_ex import sigmoid
import matplotlib.pyplot as plt

def l1_reg(l,w,dy=False):
    if dy:
        return l*abs(w)
    else: 
        return (l/2)*np.exp(-l*abs(w))

# creating random data to demonstrate
N = 50
D = 50 

# subtract 0.5 to center around 0 
X = (np.random.random((N,D)) - 0.5) * 10

# only first 3 dimension affect the output and the rest of the 47 do not
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

# generate targets
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

# perform gradient descent
costs = []
# randomly initialise weights
w = np.random.randn(D)/np.sqrt(D)

lr = 0.001
l1 = 4.996 # try different values to see results

for t in range(5000):
    Yhat = sigmoid(X.dot(w))
    delta = Yhat - Y
    w = w - lr*(X.T.dot(delta) + l1*np.sign(w) )

    cost = -(Y*np.log(Yhat)+ (1-Y)*np.log(1 - Yhat).mean() + l1_reg(l1,w,True).mean())
    costs.append(cost)

plt.plot(costs)
plt.show()


plt.plot(true_w, label='true w')
plt.plot(w, label='w map')
plt.legend()
plt.show()

