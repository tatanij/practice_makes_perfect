import numpy as np
import matplotlib.pyplot as plt

# loading data 

X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',') #data in CSV file is separated by ','
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

# plot

plt.scatter(X,Y)
plt.show()

# Calculate a and b from equations
denominator = X.dot(X) - X.mean() * X.sum() 
a= (X.dot(Y)- Y.mean()*X.sum())/denominator # since the sum of two vectors is equalivalent to the dot product. 
b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y))/denominator

#calulate predicted Y 
Yhat = a*X + b

#plot
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

# calculating r-squared

d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2) # since the sum is the equal of the dot product
print("the r-squared is: ", r2)
