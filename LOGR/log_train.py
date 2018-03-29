import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from pre_processor import get_binary_data
from learning import cross_entropy_loss
from learning import logr_ex

X, Y = get_binary_data()

# shuffle so that it's not in order
X, Y = shuffle(X,Y)

# create train and test sets
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

# initialise data
D = X.shape[1]
W = np.random.randn(D)
b = 0 

def forward(X,W,b):
    return logr_ex.sigmoid(X.dot(W)+b)

def classifcation_rate(Y,P):
    return np.mean(Y == P)

train_costs = []
test_costs = []
lr = 0.001

# do set 
for i in range(10000):
    # in each iteration, calculate the predicted train value
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy_loss.cross_entropy(Ytrain,pYtrain)
    ctest = cross_entropy_loss.cross_entropy(Ytest,pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    #apply gradient descent
    W -= lr * Xtrain.T.dot(pYtrain - Ytrain)
    b -= lr * (pYtrain - Ytrain).sum()

    # print every 1k steps
    if i % 1000 == 0:
        print(i, ctrain, ctest)


print('Final train classfication_rate:', classifcation_rate(Ytrain, np.round(pYtrain)))
print('Final test classfication_rate:', classifcation_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label='training cost')
legend2, = plt.plot(test_costs, label='testing cost')
plt.legend([legend1, legend2])
plt.show()