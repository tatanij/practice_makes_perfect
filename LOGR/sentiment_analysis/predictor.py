

from .logistic_regressor import LogisticRegressor
from .pre_process import get_split_data, word_index

Xtrain,Xtest,Ytrain,Ytest = get_split_data()

model = LogisticRegressor(Xtrain,Xtest,Ytrain,Ytest,lr=0.001,l=2.0,epochs=10)
model.model()
model.train()
model.fit()
print('Classification rate:',model.score())

threshold = 0.5
for word, index in word_index.items():
    weight = model.w[index]
    if weight > threshold or weight < -threshold:
        print(word+":", weight)