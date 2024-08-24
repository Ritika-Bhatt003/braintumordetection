import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns

image_directory= 'dataset/'   #it stores the path/location where the images is stored
no_tumor_images= os.listdir(image_directory+ 'no/')   #it is use to get the list of file names in the specific directory
yes_tumor_images= os.listdir(image_directory+ 'yes/')   #its argument is the full path of the directory containing the required images 
dataset= []
label=[]

INPUT_SIZE= 64

#path= 'no0.jpg'
#print(path.split('.')[1]) -----> the output of print(path.split('.')[1]) would be 'jpg'

for image_name in no_tumor_images:
    if(image_name.split('.')[1]== 'jpg'):
        image= cv2.imread(image_directory+'no/'+image_name) # reading the image and returning a numpy array representing the image
        image= Image.fromarray(image, 'RGB') 
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= image.resize((INPUT_SIZE,INPUT_SIZE)) #resize the image so that each of them are of same size
        dataset.append(np.array(image))  #converts the resize images into a numpy array and append it to dataset 
        label.append(0) 

for image_name in yes_tumor_images:
    if(image_name.split('.')[1]== 'jpg'):
        image= cv2.imread(image_directory+'yes/'+image_name)
        image= Image.fromarray(image, 'RGB')
        image= image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset= np.array(dataset) #it converts the dataset list of numpy arrays into a numpy array 
label= np.array(label)


#now we will split our data for training and testing. We have used 80% of the data for training and 20% for testing
x_train, x_test, y_train, y_test= train_test_split(dataset, label, test_size=0.2, random_state=0)



x_train= normalize(x_train, axis=1) #divides the data into 2 parts to increase its efficiency, so that when an unseeen data will come the model will able to predict it 
x_test= normalize(x_test, axis=1)


#model building.....

model= Sequential() #add one layer at a time
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3))) #1st layer used for image processing and abstration of useful information for it 
model.add(Activation('relu')) #decides weather a neuron should be active or not (0 to 1)
model.add(MaxPooling2D(pool_size=(2,2))) #down- smapling of data by selecting the max. value in each 2X2 block

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #flatten multi dimensional imput to a single dimension array
model.add(Dense(64)) #each neuron is connected to every other neuron in the previous layer
model.add(Activation('relu'))
model.add(Dropout(0.5)) #prevents overfitting
model.add(Dense(1)) #we are using binary classification
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history= model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test),
           shuffle=False) #our model is trained using the fit method with the training data
model.save('BrainTumor10Epochs.h5') # This file can be loaded later for inference or further training.

#graph

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()