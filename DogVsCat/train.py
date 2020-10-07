# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:55:03 2020

@author: dthakur
"""

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np
import cv2 
import os 
from random import shuffle 

TRAIN_DIR = r'C:\swMine\ML\data\train'

IMG_SIZE = 50
IMG_CHANNEL = 3

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return 1
    elif word_label == 'dog': return 0

def createTrainData():
    trainingData = []
    for img in os.listdir(TRAIN_DIR):
        label = label_img(img)
        image_path = os.path.join(TRAIN_DIR,img)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(IMG_SIZE, IMG_SIZE) )
        trainingData.append([np.array(image/255.0), label])
    shuffle(trainingData)
    np.save('trainingData.npy',trainingData)
    return trainingData

import tensorflow as tf



def createModelCompileFit():
    modelConv = tf.keras.Sequential()
    modelConv.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,IMG_CHANNEL)))
    modelConv.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelConv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    modelConv.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelConv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    modelConv.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelConv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    modelConv.add(tf.keras.layers.Flatten())
    modelConv.add(tf.keras.layers.Dense(64, activation='relu'))
    modelConv.add(tf.keras.layers.Dense(2))
    modelConv.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    modelConv.summary()
    modelConv.fit(X, Y, epochs=10)
    
    
    modelConv.save("modelConv")
    return modelConv


def evaluateModel():
    modelConv = createModelCompileFit()
    test_loss, test_acc = modelConv.evaluate(testX,  testY, verbose=2)
    print('\nTest accuracy:', test_acc)

trainData = createTrainData()
train = trainData[:-1000]
test = trainData[-1000:]

X = np.array([i[0] for i in train])
Y = np.array([i[1] for i in train])

testX = np.array([i[0] for i in test])
testY = np.array([i[1] for i in test])


evaluateModel()






        

        
    
