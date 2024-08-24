import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model= load_model('BrainTumor10Epochs.h5')
image= cv2.imread('C:\\Users\\RITIKA\\OneDrive\\Desktop\\BrainTumorDetection\\pred\\pred0.jpg')
img= Image.fromarray(image)
img= img.resize((64, 64))
img= np.array(img)
print(img)
input_img= np.expand_dims(img, axis=0) #it is preparing the image for input to a neural network by adding an extra dimension to the image array. 
result = int(model.predict(input_img))
print(result)
