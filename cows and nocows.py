from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#file = "C:\\Users\\rajes\\OneDrive\\Desktop\\computer vision\\base data\\training\\abnormal\\c268.jpg"
#image = cv2.imread(file)
# img = "C:\Users\rajes\OneDrive\Desktop\computer vision\base data\training\abnormal\c3.jpg"image.load_img("C:\Users\rajes\OneDrive\Desktop\computer vision\base data\training\abnormal\c3.jpg")
#plt.imshow(image)
#plt.show()
#var = cv2.imread("C:\\Users\\rajes\\OneDrive\\Desktop\\computer vision\\base data\\training\\abnormal\\c268.jpg")

#print(var)
train = ImageDataGenerator(rescale =1/255)
validation = ImageDataGenerator(rescale =1/255)
train_dataset = train.flow_from_directory("C:\\Users\\rajes\\OneDrive\\Desktop\\computer vision\\frames\\training",
                                         target_size=(200,200),batch_size=3,class_mode='binary')
validation_dataset = validation.flow_from_directory("C:\\Users\\rajes\\OneDrive\\Desktop\\computer vision\\frames\\validation",
                                         target_size=(200,200),class_mode='binary')
x =train_dataset.class_indices
print(x)
model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(16,(3,3),activation ='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3),activation ='relu'),
tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer =RMSprop(lr =0.0001),metrics=['accuracy'])
history = model.fit(train_dataset,epochs=8, validation_data= validation_dataset)
y=model.summary()
print(y)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.title('model_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'], loc='upper left')
plt.show()

im2=cv2.imread("frame9.jpg")
im2=cv2.resize(im2, (200,200)) # resize to 180,180 as that is on which model is trained on
print(im2.shape)
img2 = tf.expand_dims(im2, 0) # expand the dims means change shape from (180, 180, 3) to (1, 180, 180, 3)
print(img2.shape)

predictions = model.predict(img2)
#score = tf.nn.softmax(predictions[0]) # # get softmax for each output

print(predictions)
if predictions==[[1.]]:
    print("nocows")
else:
    print("cows")
#get the np.argmax, means give me the index where probability is max, in this case it got 29. This answers the response
#you got from your instructor. that is "greatest weight"

import tkinter as tk
