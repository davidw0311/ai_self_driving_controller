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
import random
def pixelate(input, rand=True):
    height, width = input.shape[:2]

    # Desired "pixelated" size
    if rand:
        a = random.randint(10,60)
        w, h = (a, a)
    else:
        w, h = (12, 12)
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
        image = cv2.imread(image_path, 1)
        
        
        pixelated = pixelate(image)
        blurred = cv2.GaussianBlur(pixelated, (29,29), 0)
        
        # uh = 255
        # us = 255
        # uv = 235
        # lh = 0
        # ls = 0
        # lv = 0
        # lower_hsv = np.array([lh,ls,lv])
        # upper_hsv = np.array([uh,us,uv])
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # thresh = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        thresh = cv2.inRange(blurred, (220,0,0),(255,110,110))
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # hull = cv2.convexHull(contours)
        # empty = np.full_like(image, 0)
        # cv2.drawContours(empty, contours, -1, (255,255,255), -1)
        thresh = ~thresh
        # blurred = cv2.GaussianBlur(thresh, (29,29), 0)
        pixelated = pixelate(thresh, rand=True)
        dilated = cv2.erode(pixelated, (33,33), iterations = 5)
        # cv2.imshow('thresh', dilated)
        # cv2.imshow('im', image)
        # cv2.imwrite('template.jpg', blurred)
        # cv2.waitKey(0)
        # n += 1
        # if n > 10:
            # break
        # break

        if not os.path.exists(save_path + '/' + str(char)):
            os.makedirs(save_path + '/' + str(char))
        write_path = save_path + '/' +str(char) + '/' + image_name 
        cv2.imwrite(write_path, dilated)

    # break

        # cv2.imshow('image', blurred)
        # cv2.waitKey(0)
