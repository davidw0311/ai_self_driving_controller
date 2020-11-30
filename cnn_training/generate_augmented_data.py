import numpy as np
import re
import os
from PIL import Image, ImageEnhance
from keras.preprocessing.image import ImageDataGenerator
import cv2
import random
from matplotlib import pyplot as plt

PATH = "/home/sylvia/ros_ws/src/my_controller/cnn_training/"
folders_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# from stackoverflow.com/questions/43383045/keras-realtime-augmentation-adding-noise-and-contrast
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation  = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


def blur_and_darken(img):
    #darkness = random.randint(8,10)/10 # factor between 0.5 and 1
    x = random.randint(1,20)
    ret_img = img
    if x%3 == 0 or x%3 == 1:
        ret_img = add_noise(img)
    ret_img = cv2.blur(img,(x,x))
    return ret_img


datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1, 
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.2,1.0],
            preprocessing_function=blur_and_darken
            )

aug_data_path = PATH + 'augmented_data/'

def save_augmented_photos(img, character):
    x = np.expand_dims(img, axis=0)
    i = 10
    this_path = aug_data_path + character
    # try:
    #     os.mkdir(this_path)
    # except OSError:
    #     print("creation of the directory %s failed" % this_path)

    for batch in datagen.flow(x, batch_size=1, 
                            save_to_dir=this_path, 
                            save_prefix=character):
        i += 1
        if i > 15: # 20 generates 12? images? 10 generages 1
            break

for i in range(len(folders_str)):
    folder_path = PATH + 'characters_pictures/' + str(folders_str[i]) + '/'
    files = os.listdir(folder_path)
    first_file = True
    for file in files:
        if first_file:
            first_file = False
            exact_path = folder_path +  '/' + file
            img = Image.open(exact_path)
            save_augmented_photos(img, folders_str[i])