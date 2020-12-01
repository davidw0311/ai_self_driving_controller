import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2

my_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PATH = '/home/davidw0311/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/'
my_dim_reversed = (105, 150)

def good_images_overlay():
    img = cv2.imread(PATH + 'good.jpg')
    files = os.listdir(PATH)
    for file in files:
        img = np.array(Image.open(PATH + file))
        avg_img = (avg_img + img)/2
    plt.imshow(avg_img)
    plt.show()


def cut(img):
    # P x(30:160) y(150:245)
    P =  img[150:245, 30:160]
    # ID x(165:285) y(150:245)
    ID = img[150:245, 165:285]
    # A1 x(30:85) y(290:330) 
    A1 = img[290:330, 30:85]
    # A2 x(85:135) y(290:330)
    A2 = img[290:330, 85:135]
    # N1 x(180:230) y(280:325)
    N1 = img[280:325, 180:230]
    # N2 x(230:275) y(280:325)
    return P

# img = np.array(Image.open(PATH + 'good.jpg'))

img = cv2.imread(PATH + 'good.jpg')
P = cut(img)
print(np.shape(P))
P_resized = cv2.resize(P,my_dim_reversed)

plt.imshow(P)
plt.show()
plt.imshow(P_resized)
plt.show()


