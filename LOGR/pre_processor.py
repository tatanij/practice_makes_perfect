import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    # df.head() # will print the first five lines of the file
    data = df.as_matrix()

    #split x and y. remember that [rows,cols]
    X = data[:, :-1] # everything up dto the last column
    Y = data[:, -1] # last column

    #normalise numerical columns in all rows.
    X[:,1] = (X[:,1]- X[:,1].mean())/X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean())/X[:,2].std()

    #work on categorical columns i.e. time of day
    # 1. get shape of original X and make new X:
    N, D = X.shape
    X2 = np.zeros((N, D+3))#D+3 because there are 4 different categorical values
    X2[:,0:(D-1)] = X[:,0:(D-1)] #most of x stays the same from the first to the penultimate column

    #One hot encoding for the four categories of time of day.
    #1. simple way
    # for n in range(N):
    #     t = int(X[n,D-1])    #time of day, either 0,1,2 or3
    #     X2[n,t+D-1]=1       #since you added 3 more columns to contain the categories of time of day 
    #     #using the values at the time of day, you

    # 2. Single line method
    Z = np.zeros((N,4))
    Z[np.arange(0,N),X[:,-1].astype(int)] = 1
    X2[:,-4:]=Z

    return X2, Y

# For Logistic Regression, only binary data is needed. Don't want the full dataset
def get_binary_data():
    X, Y = get_data()

    # filter to take only classes 0 and 1
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2,Y2