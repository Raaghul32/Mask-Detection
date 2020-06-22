# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:35:56 2020

@author: raagh
"""
import numpy as np
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from livelossplot import PlotLossesKeras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()
training_set = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
train = training_set.flow_from_directory("Train",target_size=(48,48),batch_size = 8,class_mode='binary')
testing_set = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
test = testing_set.flow_from_directory("Test",target_size=(48,48),batch_size = 4,class_mode='binary')
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
a= train.n//train.batch_size
b= test.n// test.batch_size
model_check= ModelCheckpoint("modelma.h5",monitor = 'val_accuracy',save_weights_only =True,mode='max')
callbacks = [PlotLossesKeras(),model_check]
#Model Fitting
history = model.fit(x=train,steps_per_epoch= a,epochs=10,validation_data= test,validation_steps=b,callbacks = callbacks)
#ModelTesting
ima = image.load_img('OIP.jpg',target_size=(48,48))
ima = image.img_to_array(ima)
testx= np.expand_dims(ima,axis=0)
print(train.class_indices)
pred = model.predict(testx)
if pred[0] > 0.5 :
    print("No Mask")
else:
    print("Masked")
#Creating Json File
model_json = model.to_json()
with open("model.json",'w')as json_file :
    json_file.write(model_json)