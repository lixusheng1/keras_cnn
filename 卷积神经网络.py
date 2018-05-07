#-*-coding:utf-8-*-

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,rmsprop,adagrad,Adadelta
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.activations import softmax,sigmoid,relu,tanh
from keras.datasets import mnist

# (x_train,y_train),(x_test,y_test)=mnist.load_data()
#
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
data=pd.read_csv('../kaggle_data/train.csv',header=0)
x_train=np.array([data.ix[i,1:] for i in range(0,len(data))])
y_train=np.array([data.ix[i,0] for i in range(0,len(data))])
data2=pd.read_csv('../kaggle_data/test.csv',header=0)
x_test=np.array([data2.ix[i,:] for i in range(0,len(data2))])

x_train=np.reshape(x_train,[x_train.shape[0],28,28,1])
x_test=np.reshape(x_test,[x_test.shape[0],28,28,1])

y_train=keras.utils.to_categorical(y_train,num_classes=10)


x_train=np.float32(x_train)
x_test=np.float32(x_test)
x_train/=255
x_test/255

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=[x_train.shape[1],x_train.shape[2],1],activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='Adadelta',loss=categorical_crossentropy,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=12)
# scores=model.evaluate(x_test,y_test,batch_size=128)
predictions=model.predict(x_test,batch_size=128)
predictions=np.argmax(predictions,axis=1)
print("--------------")
# print("loss=",scores[0])
# print("accuracy=",scores[1])

import pandas as pd
save=pd.DataFrame({'ImageId':range(1,len(predictions)+1),'Label':predictions})
save.to_csv('kaggle_mlp.csv',sep=',',index=False)