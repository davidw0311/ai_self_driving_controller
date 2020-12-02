import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import os
import cv2
import random

my_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PATH = '/home/sylvia/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/'
my_dim_rev = (105, 150)

def good_images_overlay():
    avg_img = np.array(Image.open(PATH + 'good.jpg'))
    files = os.listdir(PATH)
    out = None
    n=0
    for file in files:
        img = Image.open(PATH + file)
        # img = ImageOps.grayscale(img)
        img = np.array(img)
        if out is None:
            out = img
        else:
            out += img
        
        n+=1
        # avg_img = (avg_img + img)/2
    avg = np.int32(out/n)
    plt.imshow(avg)
    plt.show()


def cut(img):
    # P x(30:160) y(150:245)
    P =  img[150:245, 30:160]
    P = cv2.resize(P,my_dim_rev)
    # ID x(165:285) y(150:245)
    ID = img[150:245, 165:285]
    ID = cv2.resize(ID,my_dim_rev)
    # A1 x(20:76) y(280:330) 
    A1 = img[280:330, 20:76]
    A1 = cv2.resize(A1,my_dim_rev)
    # A2 x(76:135) y(280:330)
    A2 = img[280:330, 76:135]
    A2 = cv2.resize(A2,my_dim_rev)
    # N1 x(180:227) y(280:330)
    N1 = img[280:330, 180:227]
    N1 = cv2.resize(N1,my_dim_rev)
    # N2 x(227:280) y(280:330)
    N2 = img[280:330, 227:280]
    N2 = cv2.resize(N2,my_dim_rev)
    return P, ID, A1, A2, N1, N2

def show_cut_imgs():
    img = np.array(Image.open(PATH + 'good.jpg'))
    P, ID, A1, A2, N1, N2 = cut(img)
    #print(np.shape(P))
    # P_resized = cv2.resize(P,my_dim_reversed)
    for im in cut(img):
        plt.imshow(im)
        plt.show()
    
    #plt.imshow(P_resized)
    #plt.show()

aug_data_path = "/home/sylvia/ros_ws/src/my_controller/cnn_training/augmented_data/"
dev_set_path = '/home/sylvia/ros_ws/src/my_controller/cnn_training/dev_set/'

def delete_files(path):
    aug_data_path = "/home/sylvia/ros_ws/src/my_controller/cnn_training/augmented_data/"
    for i in range(len(my_str)):
        folder_path = path + str(my_str[i]) + '/'
        files = os.listdir(folder_path)
        for file in files:
            exact_path = folder_path + file
            if os.path.exists(exact_path):
                os.remove(exact_path)
            else:
                print('file does not exist')

def generate_dev_set():
    dev_set_path = '/home/sylvia/ros_ws/src/my_controller/cnn_training/dev_set/'
    # for i in my_str:
    #     this_path = dev_set_path + i + '/'
    #     try:
    #         os.mkdir(this_path)
    #     except OSError:
    #         print("creation of the directory %s failed" % this_path)

    good_path = '/home/sylvia/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/'
    for file in os.listdir(good_path):
        img_path = good_path + file
        img = np.array(Image.open(img_path))
        P, ID, A1, A2, N1, N2 = cut(img)
        imgs = [P, ID, A1, A2, N1, N2]
        for i,im in enumerate(imgs):
            plt.imshow(im)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            char = raw_input("what LETTER or number?\n")
            this_path = dev_set_path + str(char) + '/'
            print(str(char))

            save_im = Image.fromarray(im)
            img_name = str(char) + '_' + file.replace('.jpg','') + '_' + str(i) + '.png'
            this_path = this_path + img_name
            print(this_path)
            save_im.save(this_path, 'PNG')

#generate_dev_set()


delete_files(aug_data_path)

# im = np.array(Image.open(
#     '/home/sylvia/ros_ws/src/my_controller/cnn_training/augmented_data/1/1_0_1338.png'))
# #im = cv2.imread('/home/sylvia/ros_ws/src/my_controller/cnn_training/augmented_data/1/1_0_59.png')
# im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
# im[:,:,2] = 80 # changes the V value (between 0, 255)
# im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
# img = np.array(im)
# #cv2.imshow('img', im)

# # print(img)
# plt.imshow(img)
# plt.show()

print(my_str[15]) # index of P is 15