import numpy as np 

A = input("Enter array with values separated by commas ','.")
A = np.array(A.split(',')).astype(int)

inpt = int(input("Enter '1' for Search or '2' for Sort: \n"))
if(inpt == 1):
    x = int(input("Enter value to search for... \n"))
