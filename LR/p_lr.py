'''
linear regression means that the weight parameters are linear. 

Always plot your data before trying to fit a model. 
'''

import numpy as np
import matplotlib.pyplot as plt

#load data
X = []
Y = []

for line in open('data_poly.csv'):
    x,y = line.split(',')
    x = float(x) #rmb this is a scalar
    X.append([1, x, x*x]) #we want the x^2 term so we will treat like you do in MLR
    Y.append(float(y))

# convert the arrays to numpy arrays

X = np.array(X)
Y = np.array(Y)

'''
# see what data looks like. 
plt.scatter(X[:,1], Y) # we want only the x value so we want the second column
plt.show()
'''

#calculate weights. Just as done in mlr. Only difference in this is how the input table X is created. 
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X,w)

#plot together. 
plt.scatter(X[:,1], Y)
# plt.plot(X[:,1], Yhat) #plot prediction as a line. 

# ensures that the plot is monotically increasing. plot goes wild otherwise. 
plt.plot(sorted(X[:,1]), sorted(Yhat)) #uncomment the line above to see what I mean... 
plt.show()

#compute r-squared. 
d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("the r-squared of this model is",r2)