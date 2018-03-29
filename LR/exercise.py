import matplotlib.pyplot as plt
import numpy as np
from pandas import read_excel as rdr 

'''
The data (X1, X2, X3) are for each patient.
X1 = systolic blood pressure
X2 = age in years
X3 = weight in pounds

Therefore we can assume that X1 is the output i.e Y
'''

df = rdr('mlr02.xls')
X = df.as_matrix()


#for line in df: 
#observing relationship between 
plt.scatter(X[:,1],X[:,0])
plt.show()


plt.scatter(X[:, 2], X[:,0])
plt.show()

df['xtra'] = np.random.randn()
df['ones'] = 1 #creating bias vector
Y = df['X1']
X = df[['X2','X3', 'ones']]

#doing three linear regressions. 
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X,Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)

    #compute r^2
    d1 = Y- Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2

print("r2 for x2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X,Y))