# Importing Libraries
from distutils.dir_util import copy_tree
import os
import shutil
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras 
from keras import layers
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint


# Hyperparameters
BATCH_SIZE = 5
EPOCHS = 45
SPLIT = 0.25
IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Import
data_path = 'ColorClassification'
print(os.listdir(data_path))

os.mkdir('data')

for folder in os.listdir(data_path):
    if (folder[0].isupper() == True) and len(folder) <=16 :
        os.mkdir(f'data/{folder}')
        for file in os.listdir(f'{data_path}/{folder}'):
            shutil.copy2((f'{data_path}/{folder}/{file}'), (f'data/{folder}'))
    
    elif folder[0].isupper() != True:
        pass


# Data Preprocessing
data_path = 'data/'
classes = os.listdir(data_path)

X = []
Y = []

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)
X = np.asarray(X)
Y = pd.get_dummies(Y).values


# Taining & Test Sets Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 42, test_size= SPLIT)


# Creating model based on MobileNet
base_model = keras.applications.MobileNet(input_shape= IMG_SHAPE,
                                          include_top= False,
                                          weights= 'imagenet',
                                          pooling= 'max'
                                          )

model = keras.Sequential([
    base_model,
    layers.BatchNormalization(),
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(rate= 0.2),
    layers.Dense(9, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              metrics= ['accuracy'],
              loss= 'categorical_crossentropy'
              )


# Creating Model Callbacks
checkpoint = ModelCheckpoint('output/model_weights.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= True
                             )


# Training Model

model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          callbacks= checkpoint,
          validation_data= (X_test, Y_test),
          verbose= 1
          ) # Our validation accuracy is around 0.6 which is pretty good for the dataset that consists only out of 115 images ;). However accuracy can be improved further more with tuning hyperparameters.
