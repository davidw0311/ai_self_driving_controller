import numpy as np
import re
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import random

PATH = "/home/davidw0311/ros_ws/src/my_controller/cnn_training/"

template_path = PATH + 'augmented_data'
save_path = PATH + 'modified_augmented_data'

def pixelate(input):
    height, width = input.shape[:2]

    # Desired "pixelated" size
    w, h = (16, 16)
    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    return output    

# trial = cv2.imread('/home/davidw0311/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/good.jpg', 0)
# cv2.imshow('trial', trial)

for char in os.listdir(template_path):
    n = 0
    for image_name in os.listdir(template_path + '/' + char):
        image_path = template_path + '/' + char + '/' + image_name
        image = cv2.imread(image_path, 0)

        pixelated = pixelate(image)
        blurred = cv2.GaussianBlur(pixelated, (5,5), 0)
        (h,w) = blurred.shape
        overlay = np.random.rand(h,w)
        overlay = (np.full_like(blurred, 255) - overlay * 50).astype(np.uint8)

        darkened = cv2.addWeighted(overlay, 0.2, blurred, 0.8, 0)

        # cv2.imshow('dark', darkened)
        # cv2.waitKey(0)

        if not os.path.exists(save_path + '/' + str(char)):
            os.makedirs(save_path + '/' + str(char))
        write_path = save_path + '/' +str(char) + '/' + image_name 
        cv2.imwrite(write_path, blurred)


        # cv2.imshow('image', blurred)
        # cv2.waitKey(0)
