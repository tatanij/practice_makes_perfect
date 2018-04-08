import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import import_ipynb
from perceptron_intuition import Perceptron
import sys
sys.path.append('../functions/')
from utils import get_mnist_data, generate_xor


X,Y = get_mnist_data()

# perceptron is only capable of binary classification
# therfore only take samples where Y==0 and Y==1 then
# change these values to -1 and +1 
idx = np.logical_or(Y==0,Y==1)
X,Y = X[idx],Y[idx]
Y[Y==0] == -1

model = Perceptron()
t0 = datetime.now()
model.fit(X,Y,lr=10e-3)
print('MNIST train accuracy:',model.score(X,Y))

print('\n XOR results:')
X,Y = generate_xor(2,2)
model.fit(X,Y)
print('XOR accuracy:',model.score(X,Y))

