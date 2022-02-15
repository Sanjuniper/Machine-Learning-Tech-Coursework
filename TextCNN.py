import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import re
import csv
import string
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.word2vec import Word2Vec
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

train_data=pd.read_csv('Google Tran Tweet.csv')
test_data=pd.read_csv('Google Tran Test Tweet.csv')


train_data=shuffle(train_data)
label=train_data['label']
train_label=[]
for data in label:
    if data=='real':
        train_label.append(1)
    else:
        train_label.append(0)


def encodLabel(data):
    listLabel=[]
    for lable in data['label']:
        #print(lable)
        if(lable not in ['real','fake']):
            lable='fake'
        listLabel.append(lable)
    le =LabelEncoder()
    resultLable=le.fit_transform(listLabel)
    return resultLable

trainLable=encodLabel(train_data)
testLable=encodLabel(test_data)
print(test_data['label'])

def GetTweet(data):
    tweetdata=[]
    for tweet in data['TweetTrans']:
        tweetdata.append(str(tweet))
    return tweetdata
traindata=GetTweet(train_data)
testdata=GetTweet(test_data)
#print(len(traindata+testdata))
def read_stopwords():
    list = pd.read_csv('stopwords.txt')
    stopwords=[]
    for i in list.values:
        stopwords.append(i)
    return stopwords


def wordCut(Tweet):
    Mat=[]
    #stw = read_stopwords()
    for tweet in Tweet:
        seten=[]
        fenci=jieba.lcut(tweet)
        #stc=deleteStop(fenci)
        for d in fenci:
            if d not in [' ','\\',r'\\']:
                seten.append(d)
        Mat.append(seten)
        #print(seten)
    return Mat



traincut=wordCut(traindata)
testcut=wordCut(testdata)
Wordcut=traincut+testcut
#print(len(traindata),len(trainLable))
tokenizer=Tokenizer()
tokenizer.fit_on_texts(traincut)
vocab=tokenizer.word_index
print(len(vocab))
tokenizer=Tokenizer()
tokenizer.fit_on_texts(testcut)
vocab=tokenizer.word_index
print(len(vocab))
maxLen=30
trainID=tokenizer.texts_to_sequences(traincut)
testID=tokeni3zer.texts_to_sequences(testcut)


trainSeq=sequence.pad_sequences(trainID,maxlen=maxLen)
testSeq=sequence.pad_sequences(testID,maxlen=maxLen)
print(trainSeq)
trainCate=to_categorical(trainLable,num_classes=2)
print(trainCate)
print(trainLable)
print(testLable)
testCate=to_categorical(testLable,num_classes=2)

#train word vector
num_featurers=25
min_word_count=3
num_workers=4
context=4

model=Word2Vec(Wordcut,workers=num_workers,vector_size=num_featurers,min_count=min_word_count,window=context)
model.init_sims(replace=True)
model.save("CNNword2vecmodel")
print(model)
w2v_model=Word2Vec.load("CNNword2vecmodel")

embedding_matric=np.zeros((len(vocab)+1,25))

for word,i in vocab.items():
    try:
        embedding_vector=w2v_model.wv[str(word)]
        print('word',str(word), 'embedding',embedding_vector)
        embedding_matric[i]=embedding_vector

    except KeyError:
        continue
print(vocab.items())
print('train_data',embedding_matric.shape)
print(embedding_matric)
data=embedding_matric
Y = train_label
print(len(Y), data.shape)
#

embedder=Embedding(len(vocab)+1,100,input_length=maxLen,weights=[embedding_matric],trainable=False)
# print(embedder(trainSeq[0]))

X=trainSeq

data_2 = LocallyLinearEmbedding(n_components=2, n_neighbors = 30).fit_transform(X)

plt.figure(figsize=(8, 4))

plt.title("sklearn_LocallyLinearEmbedding")
plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)
plt.savefig("MDS_1.png")
plt.show()

# main_input=Input(shape=(maxLen,),dtype='float64')
embedder=Embedding(len(vocab)+1,100,input_length=maxLen,weights=[embedding_matric],trainable=False)
print(embedder[trainSeq])
model=Sequential()
model.add(embedder)
model.add(Conv1D(256,3,padding='same',activation='relu'))
model.add(MaxPool1D(maxLen-5,3,padding='same'))
model.add(Conv1D(32,3,padding='same',activation='relu'))
#model.add(MaxPool1D(maxLen-5,3,padding='same'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# history=model.fit(trainSeq,trainCate,batch_size=128,epochs=18,validation_split=0.1)
# model.save("TextCNN2")
# print(history)

#Evaluate
TextCNN=load_model('TextCNN2')
print(TextCNN.summary())
result=TextCNN.predict(testSeq)
print(result)
print(np.argmax(result,axis=1))
score=TextCNN.evaluate(testSeq,testCate,batch_size=32)

print(score)
