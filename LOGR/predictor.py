import numpy as np
from pre_processor import get_binary_data
from functions.activation_functions import sigmoid as s

X, Y = get_binary_data()

# Get dimensionality to initialise weights of log reg model
D = X.shape[1]
W = np.random.randn(D) # randomly initialise weights
b = 0 # bias is a scalar

def forward(X,W,b):
    return s(X.dot(W)+b)

P_Y_given_X = forward(X,W,b) 
predictions = np.round(P_Y_given_X)

# takes in targets and predictions and returns the mean number of 
# times the classification is correct
def classification_rate(Y,P):
    return np.mean(Y==P)

print("Score:", classification_rate(Y,predictions))