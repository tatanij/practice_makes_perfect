from sorting.quicksort import quickSorter as qs
from sorting import inputs
import math 

l = inputs.A
x = inputs.x
def binSearch(l,x):
    
    n = l.size
    A = qs(l,0,n-1) #sorted using quick sort algorithm
    print("Array:", A)
    i=0
    first = i
    last = n-1
    found = False
    while first != last and not found:
        indx = math.floor((first + last)/2)
        if (x == A[indx]):
            found == True
            return indx
        elif(A[indx] > x):
            last = indx -1
        else:
            first = indx
    if not found:
        indx = 0
        return indx

print("index of",x," =",binSearch(l,x))