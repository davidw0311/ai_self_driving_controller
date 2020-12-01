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
    VARIABILITY = 80
    deviation  = VARIABILITY*random.triangular(0,1,0.8)
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


def pixelate_arr(img):
    im = Image.fromarray(img)
    small_im = im.resize((32,32),resample=Image.BILINEAR)
    result = small_im.resize(im.size,Image.NEAREST)
    return np.expand_dims(result,axis=0)

def pixelate(im):
    small_im = im.resize((32,32),resample=Image.BILINEAR)
    return small_im.resize(im.size,Image.NEAREST)

def darken(img):
    x = random.triangular(80,255,90)
    ret_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ret_img[:,:,2] = x # changes the V value (between 0, 255)
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_HSV2RGB)
    return np.array(ret_img)


def blur_and_darken(img):
    x = np.int(random.triangular(1,30,20))
    ret_img = img
    # if x%3 == 0 or x%3 == 1:
    #     ret_img = add_noise(ret_img)
    ret_img = add_noise(ret_img)
    ret_img = cv2.blur(ret_img,(x,x))
    ret_img = darken(ret_img)
    ret_img = pixelate_arr(ret_img)
    return np.array(ret_img)


datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1, 
            shear_range=0.5,
            zoom_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.2,1.0],
            preprocessing_function=blur_and_darken
            )

aug_data_path = PATH + 'augmented_data/'

def save_augmented_photos(img, character):
    x = np.expand_dims(img, axis=0)
    i = 0
    this_path = aug_data_path + character
    # try:
    #     os.mkdir(this_path)
    # except OSError:
    #     print("creation of the directory %s failed" % this_path)

    for batch in datagen.flow(x, batch_size=1, 
                            save_to_dir=this_path, 
                            save_prefix=character):
        i += 1
        if i > 0: # number of pics to generate
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