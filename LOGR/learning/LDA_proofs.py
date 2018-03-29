import numpy as np

X = np.matrix([[1,7,3],[7,4,-5],[3,-5,6]])
u = X.mean(0) # col wise mean - returns row vector

# Checking if I understood how to standard deviate a vector/matrix
def std_dev(a):
    mean = np.mean(a)
    ews = a - mean # element-wise subtraction
    N = a.shape[0]
    std_dev.var = (ews.T.dot(ews))/N # variance
    std = np.sqrt(std_dev.var)
    return std

# determining how to calculate covariance 
# def covariance(a):


# calculating difference between xT(inverse(sum(x_mean))) and x_mean(inverse(sum(x)))
def matcher(x,x_mean):
    inv_sum_x = 1/(np.sum(x))
    inv_sum_x_mean = 1/(np.sum(x_mean))
    x_T_isxm = x.T.dot(inv_sum_x_mean)
    xm_T_isx = x_mean.T.dot(inv_sum_x)
    return x_T_isxm == xm_T_isx

print(matcher(X,u))