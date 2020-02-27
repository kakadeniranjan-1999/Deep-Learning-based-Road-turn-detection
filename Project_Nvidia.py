# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, Cropping2D, Lambda
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import cv2
import numpy as np
import os
import datetime
import tensorflow as tf
directory1=r'E:\Raspberry pi\Codes for practicals\Self Driving Car\Mixed_dataset\Centre\\'
v=os.listdir(directory1)
directory2=r'E:\Raspberry pi\Codes for practicals\Self Driving Car\Mixed_dataset\Left\\'
w=os.listdir(directory2)
directory3=r'E:\Raspberry pi\Codes for practicals\Self Driving Car\Mixed_dataset\Right\\'
z=os.listdir(directory3)
X=[]
C=[]
L=[]
R=[]
print('Data preprocessing started')
#for i in v[1:3226]:
#    s=directory1+str(i)
#    im=cv2.imread(s)
#    k=cv2.resize(im,(224,224))
#    print(s)
#    C.append(k)
#for j in w:
#    s=directory2+str(j)
#    im=cv2.imread(s)
#    k=cv2.resize(im,(224,224))
#    print(s)
#    C.append(k)
#for m in z:
#    s=directory3+str(m)
#    im=cv2.imread(s)
#    k=cv2.resize(im,(224,224))
#    print(s)
#    C.append(k)
#C=np.array(C)
#X.append(C)
#C=[]
#np.save('Training_Images_2.npy',X)
X=np.load('Training_Images.npy')
#print(X)
print('Done with image preprocessing')

#y=[]
#c=np.array(['Centre']*1000)
#l=np.array(['Left']*1000)
#r=np.array(['Right']*1000)
#y.append(c)
#y.append(l)
#y.append(r)



#y=np.repeat(['Centre','Left','Right'],3225,axis=0)
##print(y)
#np.save('Training_Labels.npy',y)
y=np.load('Training_Labels.npy')
print('Done with label processing')


from sklearn.preprocessing import LabelEncoder
labelencoder_y_1 = LabelEncoder()
y = labelencoder_y_1.fit_transform(y)
y = to_categorical(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[0],y,train_size =0.99, random_state=0)


#model = Sequential()
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,225,3)))
#model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu" ))
#model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu" ))
#model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu" ))
#model.add(Conv2D(64, (3,3), strides=(2,2), activation="relu" ))
#model.add(Flatten())
#model.add(Dropout(0.3))
#model.add(Dense(100))
#model.add(Dropout(0.3))
#model.add(Dense(50))
#model.add(Dropout(0.2))
#model.add(Dense(3))

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(224,224,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(3))
model.summary()

# Compiling the network with mse loss function and the adam optimizer (No accuracy matrix because it's a regression problem)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])

# For monitiring the training in tensorboard
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Training 
model.fit(X_train, y_train, validation_split=0.1,batch_size=50, epochs=2)
# Saving model
model.save('Final_Model_SR.h5')

# Training summary

#import matplotlib.pyplot as plt
#
#history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
#
## Plot training & validation accuracy values
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

y_pred=model.predict(X_test)
print(y_pred)
y_pred=y_pred.argmax(1)
y_pred=labelencoder_y_1.inverse_transform(y_pred)
print(y_pred)
y_test=y_test.argmax(1)
y_test=labelencoder_y_1.inverse_transform(y_test)
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
print('done')