import numpy as np
from random import randint

def swap(a,b):
    temp = a
    a = b
    b = temp
    return [a,b]

def partition(A,p,r):
    i = p+1
    j = r
    x = A[p]
    is_sorted = np.all(A[:-1] <= A[1:])
    while not is_sorted:
        while (i <= j and A[i] <= x):
            i+=1
        while (A[j] >= x and j>=i):
            j-=1
        if j < i:
            is_sorted = True
        else: 
            A[i],A[j]= swap(A[i],A[j])
            
        # A[i],A[j] = swap(A[i],A[j])
        
        # is_sorted = np.all(A[:-1] <= A[1:])
    A[p],A[j] = swap(A[p],A[j])
    return j

def quickSorter(l,p,r):
    if(p<r):
          splitpt = partition(l, p,r)
          
          quickSorter(l, p, splitpt-1)
          quickSorter(l, splitpt+1, r)
    return l

