import numpy as np
import re
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import random

PATH = "/home/davidw0311/ros_ws/src/my_controller/cnn_training/"
folders_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

template_path = PATH + 'augmented_data'
save_path = PATH + 'modified_augmented_data'

# def pixelate(input):
#     height, width = input.shape[:2]

#     # Desired "pixelated" size
#     w, h = (16, 16)
#     # Resize input to "pixelated" size
#     temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)
#     # Initialize output image
#     output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

#     return output    

# # trial = cv2.imread('/home/davidw0311/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/good.jpg', 0)
# # cv2.imshow('trial', trial)

# for char in os.listdir(template_path):
#     n = 0
#     for image_name in os.listdir(template_path + '/' + char):
#         image_path = template_path + '/' + char + '/' + image_name
#         image = cv2.imread(image_path, 0)

#         pixelated = pixelate(image)
#         blurred = cv2.GaussianBlur(pixelated, (5,5), 0)
        
#         if not os.path.exists(save_path + '/' + str(char)):
#             os.makedirs(save_path + '/' + str(char))
#         write_path = save_path + '/' +str(char) + '/' + image_name 
#         cv2.imwrite(write_path, blurred)


#         # cv2.imshow('image', blurred)
#         # cv2.waitKey(0)


        


def blur(img):
    x = random.randint(1,20)
    return (cv2.blur(img,(x,x)))

datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.2,1.0],
            preprocessing_function=blur
            )

aug_data_path = PATH + 'augmented_data/'

def save_augmented_photos(img, character):
    x = np.expand_dims(img, axis=0)
    i = 0
    this_path = aug_data_path + character
    
    if not os.path.exists(this_path):
        os.makedirs(this_path)

    for batch in datagen.flow(x, batch_size=1, 
                            save_to_dir=this_path, 
                            save_prefix=character):
        i += 1
        if i > 500: # 20 generates 12? images? 10 generages 1
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
