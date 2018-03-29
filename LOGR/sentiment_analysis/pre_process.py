# implemented attempt at logistic regression
from bs4 import BeautifulSoup 
import numpy as np 
import re
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('\\','/')


pos = open(dir_path+"/electronics/positive.review","r")
neg = open(dir_path+"/electronics/negative.review","r")

pos_contents = pos.read()
neg_contents = neg.read()

soup_pos = BeautifulSoup(pos_contents).find_all('review_text')
soup_neg = BeautifulSoup(neg_contents).find_all('review_text')

positive_review = np.array([i.get_text().lower().replace('\n','') for i in soup_pos])
positive_review = np.array([re.sub(r'([^\s\w]|_)+', ' ', w) for w in positive_review])

# randomise texts
np.random.shuffle(positive_review)

negative_review = np.array([i.get_text().lower().replace('\n','') for i in soup_neg])
negative_review = np.array([re.sub(r'([^\s\w]|_)+', ' ', w) for w in negative_review])


positive_review = positive_review[:len(negative_review)]

stop_words= open(dir_path+'/stopwords.txt').read().split('\n')

idx = 0

'''
# PREVIOUSLY encapsulated functionality of both word_picker and indexer

negative_words, positive_words = {}

def word_picker(word_arr, target_array, stop_words):
    if not any(word in sw for sw in stop_words) and len(word) >= 2:
        if word not in target_array:
            target_array[word] = 1
        else:
            target_array[word] +=1
    return target_array

'''

def word_picker(word_arr, target_array, stop_words):
    for word in word_arr:
        if not any(word in sw for sw in stop_words) and len(word) >= 2:
            w_a.append(word)
    return target_array.append(w_a)

def indexer(wi,ci, word_arr):
    for word in word_arr:
        if word not in wi:
            wi[word] = ci
            ci+=1

word_index = {}
positive_words = []
negative_words = []

# index positive words into word index and store in positive array
for i in positive_review:
    words = i.split(" ")
    w_a = []
    word_picker(words, positive_words, stop_words)
    indexer(word_index, idx, w_a)
    # if np.where(positive_review==i) == 0:print(w_a) # check the two arrays 

for j in negative_review:
    words = i.split(" ")
    w_a = []
    word_picker(words, negative_words, stop_words)
    indexer(word_index,idx,w_a)
    # if np.where(negative_review == i) == 0:print(w_a)

vocabulary_size = len(word_index)

# defining the inputs
def input_vectors(reviews,label):
    x = np.zeros(vocabulary_size+1)
    for words in reviews:
        i = word_index[words]
        x[i] +=1
    #normalise data
    x = x/x.sum()
    x[-1] = label
    return x 

N = len(positive_review)+len(negative_review)
data = np.zeros((N,vocabulary_size+1))
i = 0 

# apply 1 as label to positive words
for w in positive_words:
    xy = input_vectors(w,1)
    data[i,:]=xy
    i+=1

# apply 0 as label to negative words
for w in negative_words:
    xy = input_vectors(w,0)
    data[i,:] = xy
    i +=1

def get_split_data():
    #create test/strain splits
    np.random.shuffle(data)

    X= data[:,:-1]
    Y = data[:,-1]

    Xtrain = X[:-500,]
    Xtest = X[-500:,]
    Ytrain = Y[:-500,]
    Ytest = Y[-500:,]

    return Xtrain,Xtest,Ytrain,Ytest

