import numpy as np 
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self, train: tuple, hidden_units = 8, learning_rate = 0.001, l = 0.02 , batch_size = 128, epochs = 100):
        # Initialising first layer's random weights and biases
        self.__W1 = np.random.randn(train[0].shape[1],hidden_units)
        self.__b1 = np.random.randn(hidden_units)

        # Initialising second layer's random weights and biases
        self.__W2 = np.random.randn(hidden_units,hidden_units)
        self.__b2 = np.random.randn(hidden_units)
        
        # Initialising output layer's random weights and biases
        self.__W3 = np.random.randn(hidden_units)
        self.__b3 = np.random.randn(1)
        
        self.X, self.Y = train
        
        self.lr = learning_rate
        self.l = l
        self.epochs = epochs
        self.batch_size = batch_size
        
        
    def __forward__(self,X):
        # initialise training data
        self.__X1 = X.dot(self.__W1) + self.__b1
        self.__h1 = self.__sigmoid__(self.__X1)
        
        # applying random weight to first hidden layer 
        self.__X2 = self.__h1.dot(self.__W2) + self.__b2
        self.__h2 = self.__sigmoid__(self.__X2)
         
        #applying random weight to output layer
        self.__X3 = self.__h2.dot(self.__W3) + self.__b3
        self.__Y_out = self.__sigmoid__(self.__X3)
       
        return self.__Y_out, self.__h2
    
    def __diff__(self, t, y):
        return t - y
    
    def __sigmoid__(self,  z, w_n = None, b_n = None, chain_link = None, prev_weight = [], X = None,dy=False):
        if dy:
            dz = z*(1-z)
            # check for the first step of back prop in first layer.
            if len(prev_weight) <= 0:
                return (chain_link.dot(z), chain_link.sum())

            # update error's shape to propagate to previous layer
            self.__chain_link__ = np.dot(chain_link,prev_weight.T) if prev_weight.shape[1:] else np.outer(chain_link,prev_weight)
            dj_dz = self.__chain_link__*dz
            return (dj_dz.T.dot(X).T, (dj_dz).sum(axis=0))
        else:
            return 1/(1+np.exp(-z))
    
    
    def __update__(self, layer, prev_weight=[], W=None, b =None, X = None, chain_link=None):
        s = self.__sigmoid__(layer, W, b, chain_link, prev_weight, X,True)

        W += self.lr * self.__diff__(s[0], self.__l2_regularization__(self.l, W,True))
        b += self.lr * self.__diff__(s[1], self.__l2_regularization__(self.l, b,True))
        
    def __backward__(self): 
        # update weights and biases using l2 regularisation.
        self.__update__(
                        self.__Y_out, [], 
                        self.__W3, self.__b3, 
                        self.__h2, 
                        self.__diff__(self.__Y_, self.__Y_out)
                       )
        
        self.__update__(
                        self.__h2, self.__W3, 
                        self.__W2, self.__b2, 
                        self.__h1, self.__diff__(self.__Y_, self.__Y_out)
                        )

        self.__update__(
                        self.__h1, self.__W2, 
                        self.__W1, self.__b1, 
                        self.__X_, self.__chain_link__
                        )
    
    def __cross_entropy__(self, t, y, dy = False):
        if dy:
            return y-t
        else:
            return -np.mean(t * np.log(y) + (1-t) * np.log(1-y))
    
    def __l2_regularization__(self, l, w, dy = False):
        if dy:
            return l * w
        else:
            return l * np.dot(w.T, w)
        
    def train(self):
        self.__losses = []
        self.__accuracies = []
        for i in range(self.epochs*1000):
            self.__X_, self.__Y_ = shuffle(self.X, self.Y)
            self.__X_, self.__Y_ = self.__X_[:self.batch_size], self.__Y_[:self.batch_size]

            self.__forward__(self.__X_)
            loss = self.__cross_entropy__(self.__Y_, self.__Y_out)

            self.__backward__()
            prediction = self.__predict__(self.X)

            accuracy = 1 - np.abs(prediction - self.Y).mean()
            
            # keep track of training progress
            if i % 1000 == 0:
                print(
                    "Epoch #{:d}".format(int(i/1000)+1),
                    ": loss =", round(loss,4), 
                    "accuracy :", round(accuracy,4)
                )
                
                self.__losses.append(loss)
                self.__accuracies.append(accuracy)

    def visualise(self, val : str):
        val = val.lower()
        assert (val == 'accuracy' or val == 'loss'), "Requested data is not loss or accuracy."

        plt.plot(self.__losses) if val == 'loss' else plt.plot(self.__accuracies)
        plt.ylabel(val)
        plt.xlabel('epochs')
        plt.show()
        
    def __predict__(self, X):
        Y, _ = self.__forward__(X)
        return Y 
    
    # Fit test data to trained model
    def fit(self,test : tuple):
        self.Y_test = test[1]
        N, D = test[0].shape
        
        # Apply learned weights to dataset
        self.Y_hat = np.round(self.__predict__(test[0]))
    

    def score(self):
        return np.mean(self.Y_test==self.Y_hat)
        