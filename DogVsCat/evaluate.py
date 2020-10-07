# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:46:19 2020

@author: dthakur
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from random import shuffle
import cv2
import json
IMG_SIZE = 50
TEST_NUM  = 10

def createTestData(testDir):
    testData = []
    for img in os.listdir(testDir):
        image_path = os.path.join(testDir,img)
        num = img.split('.')[0]
        image = cv2.imread(image_path)
        image = cv2.resize(image,(IMG_SIZE, IMG_SIZE) )
        testData.append([np.array(image/255.0), num])
    shuffle(testData)
    np.save('testData.npy',testData)
    return testData

def printPrediction(i, predictions):
    res = np.argmax(predictions[i])
    if res == 1: print("It's a cat!")
    elif res == 0: print("It's a dog!")
    
def main():
    model = tf.keras.models.load_model("modelConv")
    config = json.load(open("config.json"))
    TEST_NUM = config['showNimages']
    probability_model = tf.keras.Sequential([model, 
                                              tf.keras.layers.Softmax()])
    TEST_DIR = config['testDir']
    testData = createTestData(TEST_DIR)
    testDataNum = [i[1] for i in testData]
    testDataX = np.array([i[0] for i in testData])
    predictions = probability_model.predict(testDataX)
    for i in range(0,TEST_NUM):
        img = mpimg.imread(TEST_DIR+r'/'+str(testDataNum[i])+r'.jpg')
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.show()
        printPrediction(i,predictions)
        
if __name__ == "__main__":
    main()
    