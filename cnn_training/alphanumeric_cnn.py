import math
import numpy as np
import re
import os

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image

# from ipywidgets import interact
# import ipywidgets as ipywidgets

from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend
from keras.callbacks import EarlyStopping 


PATH = "/home/davidw0311/ros_ws/src/my_controller/cnn_training/characters_pictures/"
folders_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

aug_data_path = "/home/davidw0311/ros_ws/src/my_controller/cnn_training/other_aug_data/"
imgset = []
for i in range(36):
    folder_path = PATH + str(folders_str[i]) + '/'
    files = os.listdir(folder_path)
    pair_list = []
    for file in files:
      exact_path = folder_path +  '/' + file
      pair_list.append([np.array(Image.open(exact_path)), i])
    aug_folder_path = aug_data_path + str(folders_str[i]) + '/'
    for file in os.listdir(aug_folder_path):
      exact_path = aug_folder_path + file
      pair_list.append([np.array(Image.open(exact_path)), i])
    imgset.append(np.array(pair_list))
    # print("Loaded {:} images from folder:\n{}".format(imgset[i].shape[0], folder_path))

my_tuple = tuple(imgset[i] for i in range(36))
all_dataset = np.concatenate(my_tuple, axis=0)
np.random.shuffle(all_dataset)
print(np.shape(all_dataset))

## Generate X and Y datasets
X_dataset_orig = np.array([data[0] for data in all_dataset[:]])
Y_dataset_orig = np.array([[data[1]] for data in all_dataset]).T
# print(Y_dataset_orig)

NUMBER_OF_LABELS = 36
CONFIDENCE_THRESHOLD = 0.01

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

  
## Normalize X (images) dataset
X_dataset = X_dataset_orig/255.

## Convert Y dataset to one-hot encoding
Y_dataset = convert_to_one_hot(Y_dataset_orig, NUMBER_OF_LABELS).T
# print(Y_dataset)

VALIDATION_SPLIT = 0.2

print("Total examples: {:d}\nTraining examples: {:f}\nTest examples: {:f}".
      format(X_dataset.shape[0],
             math.ceil(X_dataset.shape[0] * (1-VALIDATION_SPLIT)),
             math.floor(X_dataset.shape[0] * VALIDATION_SPLIT)))
print("X shape: " + str(X_dataset.shape))
print("Y shape: " + str(Y_dataset.shape))

## Display images in the training data set. 
def displayImage(index):
  plt.imshow(X_dataset[index])
  caption = ("y = " + str(Y_dataset[index]))#str(np.squeeze(Y_dataset_orig[:, index])))
  plt.text(0.5, 0.5, caption, 
           color='orange', fontsize = 20,
           horizontalalignment='left', verticalalignment='top')
  plt.show()


# interact(displayImage, 
#         index=ipywidgets.IntSlider(min=0, max=X_dataset_orig.shape[0],
#                                    step=1, value=10))

displayImage(1)

# Source: https://stackoverflow.com/questions/63435679
def reset_weights(model):
  for ix, layer in enumerate(model.layers):
      if (hasattr(model.layers[ix], 'kernel_initializer') and 
          hasattr(model.layers[ix], 'bias_initializer')):
          weight_initializer = model.layers[ix].kernel_initializer
          bias_initializer = model.layers[ix].bias_initializer

          old_weights, old_biases = model.layers[ix].get_weights()

          model.layers[ix].set_weights([
              weight_initializer(shape=old_weights.shape),
              bias_initializer(shape=len(old_biases))])

## MODEL DEFINITION
conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(150, 105, 3)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(36, activation='softmax'))

conv_model.summary()

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

# print(X_dataset.shape,Y_dataset.shape)

## Define training and validation data sets
train_end_index = np.int32(math.ceil(X_dataset.shape[0]*(1-VALIDATION_SPLIT)))
val_end_index = np.int32(math.floor(X_dataset.shape[0]*VALIDATION_SPLIT))
train_data = X_dataset[0:train_end_index]
train_target = Y_dataset[0:train_end_index]
val_data = X_dataset[0:val_end_index]
val_target = Y_dataset[0:val_end_index]

## TRAIN CNN
history_conv = conv_model.fit(train_data, train_target, 
                              validation_data=(val_data, val_target),
                              epochs=20, 
                              batch_size=16)

## Plot losses

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

## Plot accuracy

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

## Plot confusion matrix

predictions = conv_model.predict(val_data)
predictions = [np.argmax(p) for p in predictions] # gets index value from one hot encoding
print(predictions)

actual = [np.argmax(p) for p in val_target]
print(actual)

# conv_model.save('alphanumeric_detector_model') # this one is trained on unaugmented data
conv_model.save('alphanumeric_detector_model_v3') # trained on augmented data
