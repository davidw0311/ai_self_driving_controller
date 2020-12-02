import sys
import random
import os
import numpy as np

import cv2

import tensorflow as tf
from keras import models
import keras

def files_in_folder(folder_path):
    return os.listdir(folder_path)

def get_one_hot_encoding(value):
    # value should be a character either 0-9 or A-Z

    encoding = np.zeros(36)

    # number
    if ord(value) > 47 and ord(value) < 58:
        encoding[ord(value)-48] = 1
    elif ord(value) > 64 and ord(value) < 91:
        encoding[ord(value)- 65 + 10] = 1

    return encoding

def decode_one_hot(encoding):
    if encoding < 10:
        return str(encoding)
    else:
        return chr(encoding-10 + 65) 

train_path = 'train/'
val_path = 'val/'

val_folder = files_in_folder(val_path)

val_dataset = np.array([cv2.imread(val_path + filepath, 1) for filepath in val_folder]) 
val_target = np.array([get_one_hot_encoding(filepath[0]) for filepath in val_folder])

david_path = '/home/davidw0311'
cnn_path = '/ros_ws/src/my_controller/cnn_training/'
PATH = david_path + cnn_path
model_path = PATH + 'detector_model_v6_7epoch_lenet'

my_dim_rev = (105, 150)

conv_model = models.load_model(model_path, compile=True)
predictions = conv_model.predict(val_dataset)
predictions = [np.argmax(p) for p in predictions]
print(predictions)

actual = [np.argmax(p) for p in val_target]
print(actual)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual, predictions)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
df_cm = pd.DataFrame(cm, index = [i for i in LABELS], columns = [i for i in LABELS])

plt.figure(figsize=(20,14))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) # font size

plt.show()