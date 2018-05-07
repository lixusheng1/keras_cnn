#-*-coidng:utf-8-*-
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,rmsprop,adagrad
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.activations import softmax,sigmoid,relu,tanh
from keras.datasets import mnist

# (x_train,y_train),(x_test,y_test)=mnist.load_data()
#
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# x_train=np.reshape(x_train,[x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
# x_test=np.reshape(x_test,[x_test.shape[0],x_test.shape[1]*x_test.shape[2]])
data=pd.read_csv('../kaggle_data/train.csv',header=0)
x_train=np.array([data.ix[i,1:] for i in range(0,len(data))])
y_train=np.array([data.ix[i,0] for i in range(0,len(data))])
data2=pd.read_csv('../kaggle_data/test.csv',header=0)
x_test=np.array([data2.ix[i,:] for i in range(0,len(data2))])



y_train=keras.utils.to_categorical(y_train,num_classes=10)


x_train=np.float32(x_train)
x_test=np.float32(x_test)
x_train/=255
x_test/255

model=Sequential()
model.add(Dense(input_dim=28*28,units=512,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(units=512,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='rmsprop',loss=categorical_crossentropy,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=20)

predictions=model.predict(x_test,batch_size=128)
predictions=np.argmax(predictions,axis=1)

print("--------------")


import pandas as pd
save=pd.DataFrame({'ImageId':range(1,len(predictions)+1),'Label':predictions})
save.to_csv('kaggle_mlp.csv',sep=',',index=False)




