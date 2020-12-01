import os
import numpy as np
import cv2
import matplotlib.pyplot as plot
from PIL import Image

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

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


model_path = 'detector_model_v3_20epoch'
conv_model = models.load_model(model_path, compile=True)

my_dim_reversed = (105, 150)

def good_images_overlay():
    avg_img = np.array(Image.open(PATH + 'good.jpg'))
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
    
    N2 = img[230:275, 223:270]

    return P, ID, A1, A2, N1, N2

PATH = '/home/davidw0311/ros_ws/src/my_controller/cnn_training/temp_test_photos/good/'
# img = np.array(Image.open(PATH + 'good.jpg'))

space = 5

uh = 123
us = 255
uv = 228
lh = 107
ls = 102
lv = 79
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

for file in os.listdir('temp_test_photos/good'):
    img = cv2.imread(PATH + file)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh_blue = cv2.inRange(img_HSV, lower_hsv, upper_hsv)
    thresh_black = cv2.inRange(img_HSV, (0,0,0), (0,0,67))
    cv2.imshow('plate', img)

    # ret, thresh = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    total_thresh = cv2.bitwise_or(thresh_blue, thresh_black)

    cv2.imshow('thresh', total_thresh)

    im2, contours, hierarchy = cv2.findContours(total_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    empty = np.zeros_like(img)
    # empty = cv2.merge((empty, empty, empty))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    characters = []

    for c in cntsSorted:
        area = cv2.contourArea(c)
        if area < 8000 and area > 100:
            M = cv2.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            if cX > space and cY > space:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                (x1, y1), (x2, y2), (x3,y3), (x4,y4) = box
                top_left_x = min([x1,x2,x3,x4])
                top_left_y = min([y1,y2,y3,y4])
                bot_right_x = max([x1,x2,x3,x4])
                bot_right_y = max([y1,y2,y3,y4])

                
                # print('before crop', img.shape)
                # cropped_number = img[top_left_y-space:bot_right_y+space, top_left_x-space:bot_right_x+space]
                # print(cropped_number.shape)
                # cv2.imshow('cropped', cropped_number)
                cv2.rectangle(empty, (top_left_x,top_left_y), (bot_right_x,bot_right_y), (0,255,0), -1)
                
                cv2.drawContours(empty, [c], -1, (0,0,255), 0)
                # cv2.imshow('empty', empty)
                
                # cv2.waitKey(0)dddd
                
                # each character, corners and centroid
                if cX > 10 and cX < 290 and cY > 10 and cY < 390:
                    characters.append((top_left_x,top_left_y,bot_right_x,bot_right_y,cX,cY,area))

    from math import sqrt      
    # print(characters)


    i = 0
    while i < len(characters) - 1:
        j = i+1
        while j < len(characters):
            if sqrt((characters[i][4] - characters[j][4])**2 + (characters[i][5] - characters[j][5])**2) < 20:
                if characters[i][6] > characters[j][6]:
                    characters.pop(j)
                elif characters[i][6] < characters[j][6]:
                    characters[i] = characters[j]
                    chracters.pop(j)
            else:
                j+= 1
        i+=1


    print(characters)
    empty = np.zeros_like(img)
    cropped_characters = []
    kernel = np.ones((3,3),np.uint8)
    eroded_thresh = cv2.erode(total_thresh, kernel,iterations = 1)
    eroded_thresh = ~eroded_thresh
    # cv2.imshow('eroded thresh', eroded_thresh)
    cv2.waitKey(0)
    def make_3_channel(f):
        return cv2.merge((f,f,f))

    for i in characters:
        top_left_x,top_left_y,bot_right_x,bot_right_y,cX,cY,area = i
        
        cropped_number = eroded_thresh[top_left_y-space:bot_right_y+space, top_left_x-space:bot_right_x+space]
        cropped_number = make_3_channel(cropped_number)
        cropped_characters.append((cropped_number, cX, cY,area))

    cropped_characters = sorted(cropped_characters, key=lambda x: x[2])
    location = []
    digits = []
    print(len(cropped_characters), 'cropped characters')
    if len(cropped_characters) > 6:
        min_area = 200000
        while len(cropped_characters) > 6:
            for i, char in enumerate(cropped_characters):
                # char is (cropped_number, cX, cY,area)
                if char[3] < min_area:
                    min_area = char[3]
                    index = i
            cropped_characters.pop(index)

    elif len(cropped_characters) < 6:
        print('THERE WERE LESS THAN 6 CHARACTES FOUND')
    for character in cropped_characters:
        if character[2] < 200:
            location.append(character)
        else:
            digits.append(character)
        # cv2.imshow('cropped', character[0])
        # cv2.waitKey(0)
    # print(len(location), 'location')
    # print(len(digits), 'digits')
    location = sorted(location, key=lambda x: x[1])
    digits = sorted(digits, key = lambda x: x[1])
    # print(len(location))
    # print(len(digits))

    P = location[0][0]
    ID = location[1][0]
    A1 = digits[0][0]
    A2 = digits[1][0]
    N1 = digits[2][0]
    N2= digits[3][0]

    # cv2.imshow('P', P)
    # cv2.imshow('ID', ID)
    # cv2.waitKey(0)


    # cv2.imshow('img', img)
    # cv2.imshow('empty', empty)
    # cv2.waitKey(0)

    # P, ID, A1, A2, N1, N2 = cut(img)



    # print(np.shape(P))
    P = cv2.resize(P,my_dim_reversed,interpolation =cv2.INTER_AREA)
    ID = cv2.resize(ID,my_dim_reversed,interpolation = cv2.INTER_AREA)
    A1 = cv2.resize(A1,my_dim_reversed,interpolation = cv2.INTER_AREA)
    A2 = cv2.resize(A2,my_dim_reversed,interpolation = cv2.INTER_AREA)
    N1 = cv2.resize(N1,my_dim_reversed,interpolation = cv2.INTER_AREA)
    N2 = cv2.resize(N2,my_dim_reversed,interpolation = cv2.INTER_AREA)

    cv2.imshow('P', P)
    cv2.imshow('ID', ID)
    cv2.imshow('A1', A1)
    cv2.imshow('A2', A2)
    cv2.imshow('N1', N1)
    cv2.imshow('N2', N2)

    # # img = cv2.merge((img,img,img))
    def predict(c):
        for i in c:
            predictions = conv_model.predict(np.array([i]))
            prediction = np.argmax(predictions)
            print(decode_one_hot(prediction))

    predict([P,ID,A1,A2,N1,N2])



    cv2.waitKey(0)
