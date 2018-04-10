import numpy as np 
import sys

sys.path.append('../../functions')
from activation_functions import sigmoid
from loss_functions import cross_entropy
from regularisation_functions import l2_regularization

class LogisticRegressor:
    # inputs(X: 'input data', Y: 'predicants', lr: 'learning rate', l: 'lambda')
    def __init__(self, X, Xtest,Y,Ytest,lr,l,epochs):
        self.X = X
        self.Xtest = Xtest
        self.Y = Y
        self.Ytest = Ytest
        self.lr = lr
        self.l = l
        self.epochs = epochs

    def model(self):
        self.N,self.D= self.X.shape
        self.N_t, self.D_t = self.Xtest.shape
        ones = np.array([[1]*self.N]).T
        ones_t = np.array([[1]*self.N_t]).T
        self.Xb = np.concatenate((ones,self.X), axis=1)
        self.X_t_b = np.concatenate((ones_t,self.Xtest), axis=1)
        self.w = np.random.randn(self.D+1)
        z = self.Xb.dot(self.w)
        self.Yhat = sigmoid(z)

    def train(self):
        for i in range(self.epochs*100):
            if i % 100 == 0:
                print(cross_entropy(self.Y,self.Yhat))
            
            # update weights
            self.w += self.lr * (np.dot((self.Y-self.Yhat).T,self.Xb) - l2_regularization(self.l, self.w, True) )
            self.Yhat = sigmoid(self.Xb.dot(self.w))

        print('Final w:',self.w)

    def fit(self):
        z_test = self.X_t_b.dot(self.w)
        self.Y_t_hat = np.round(sigmoid(z_test))

    def score(self):
        return np.mean(self.Ytest==self.Y_t_hat)
            
