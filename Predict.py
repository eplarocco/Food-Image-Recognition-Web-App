# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:02:19 2022

@author: Eileanor
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predict (Classify New Images)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Category Names
cat = os.listdir("./output/test")

#Load and prepare the image
def load_image(filename):
	img = load_img(filename, target_size=(180, 180)) # load the image and resize
	img = img_to_array(img) # convert to array
	img = img.reshape(1, 180, 180, 3) # reshape into a single sample with 3 channels (rgb)
	return img
 
#Predict the image class
def run_example(filepath):
    img = load_image(filepath) # load the image
    model = load_model('CNN.model') # load model
    #Predict the class
    result = model.predict(img)[0]
    result = result.tolist()
    index = result.index(max(result))
    print(cat[index])
 
#Entry point, run the example
run_example('./Test/test.jpg')
run_example('./Test/test1.jpg')
