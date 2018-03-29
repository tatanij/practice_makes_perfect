import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#load data
X=[]
Y=[]

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')

    #we want to have a bias term but it is implicit in solution. .: create x0 and say it's equivalent to 1 all the time
    X.append([1, float(x1), float(x2)]) 
    Y.append(float(y))

#convert X and Y into numpy arrays. 
X= np.array(X)
Y = np.array(Y)

#plot the data. 
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
# plt.show()
'''

#calculate weights
#Won't be inverting X tranpose X, we will just use the np linalg solver
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y)) #use dot because the multiply function does an element wise multiplication as opposed to a matrix mult.
Yhat = np.dot(X, w) #RMB X.T is an nxd matrix where each sample is a row, it's easier to do X * w. 

#compute r-squared. 
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print("the r-squared is: ", r2)
