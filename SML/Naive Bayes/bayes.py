import numpy as np
from utils import get_mnist_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_norm as mvn

class Bayes(object):
    def fit(self, X,Y):
        N,D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gausians[c] = {
                'mean': current_x.mean(axis=0)
                'cov': np.cov(current_x.T) + np.eye(D) + smoothing
            }
            self.priors[c] = float(len(Y[Y==c]))/len(Y)
    
    
    def predict(self,X):
        N,D = X.shape
        K = len(self.gaussians)
        Yhat = np.zeros((N,K))
        for c, n in self.gaussians.items():
            mean, cov = n['mean'],n['cov']
            Yhat[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.self.priors[c]
            
        return np.argmax(Yhat,axis=1)
    
    def score(self,X,Y):
        Yhat = self.predict(X)
        return np.mean(Y==Yhat)
    