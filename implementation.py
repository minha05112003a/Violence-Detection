"""
Created on Sat Feb 29 14:38:11 2020
@author: pratt
"""
import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
import pickle
import cv2 
import math
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import tkinter
import PIL.ImageTk
from tkinter import *  
from PIL import ImageTk,Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
import keras as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
seed = 78
test_size = 0.33
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import keras as k
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import MaxPooling2D
import cv2
import math
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.layers import Dense, Activation
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
seed = 78
test_size = 0.33
import cv2
from PIL import Image, ImageDraw, ImageFont
import time
from numpy import loadtxt
from xgboost import XGBClassifier
loaded_model =    pickle.load(open("C:/Users/pratt/Desktop/ViolenceNonviolence.dat", 'rb'))

#%%
import cv2 
import numpy as np 

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('C:/Users/pratt/Downloads/test7.mp4') 
# Check if camera opened successfully 
if (cap.isOpened()== False):  
    print("Error opening video  file") 
i =0  
y_temp= []
array=[]
deno = 0
sum=0

while(cap.isOpened()): 
    
    frameId = cap.get(1)  
    ret, frame = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    if ret == True: 
        font = cv2.FONT_HERSHEY_SIMPLEX 
    if (ret != True):
        break
    if (frameId):
        X_temp = cv2.resize(frame, (64,64))
        
        
        X_temp = np.reshape(X_temp, (1, 64*64*3))
        X_temp = X_temp/255
        
        y_temp=loaded_model.predict(X_temp)
        sum=y_temp
        array.append(y_temp)
        """
        p = 1
        for k in range (1,6):
            if (i-k) >= 0:
                sum=sum + array[i-k]
                p=p+k
        
        sum = sum/p"""
        if sum<0.5 :
            y_temp = 0
        else:
            y_temp = 1
        
    
    if (y_temp == 0):
        
        cv2.putText(frame,  
                    'NON VIOLENCE',  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)
    if (y_temp == 1):
        cv2.putText(frame,  
                    'VIOLENCE',  
                    (50, 50),  
                    font, 1,  
                    (0, 255, 255),  
                    2,  
                    cv2.LINE_4)
    print(y_temp)
    cv2.imshow('Frame', frame) 
    i = i+1
    
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break
  
cap.release() 

# Closes all the frames 
cv2.destroyAllWindows()

#%%

