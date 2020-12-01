import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

train_folder = files_in_folder(train_path)
val_folder = files_in_folder(val_path)

# cv2 imread (img, 0) reads the image as grayscale
train_dataset = np.array([cv2.imread(train_path + filepath, 1) for filepath in train_folder]) 
train_target = np.array([get_one_hot_encoding(filepath[0]) for filepath in train_folder])

print(train_dataset[0].shape)

val_dataset = np.array([cv2.imread(val_path + filepath, 1) for filepath in val_folder]) 
val_target = np.array([get_one_hot_encoding(filepath[0]) for filepath in val_folder])

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

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

INPUT_SHAPE = (150,105,3)
OUTPUT_SHAPE = 36
conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=INPUT_SHAPE))
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
conv_model.add(layers.Dense(OUTPUT_SHAPE, activation='softmax'))

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

history = conv_model.fit(
    train_dataset,
    train_target,
    batch_size=32,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch  
    validation_data=(val_dataset, val_target),
)

conv_model.save('detector_model_v3_nothing')

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

