from keras import models
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

my_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PATH = '/home/davidw0311/ros_ws/src/my_controller/cnn_training/'
model_path = PATH + 'alphanumeric_detector_model'

conv_model = models.load_model(model_path, compile=True)
my_dim_rev = (105, 150)

#img_path = PATH + 'characters_pictures/J/plate_JP69_0.png' # J
#img_path = PATH + 'characters_pictures/E/plate_EB49_0.png' # E
# img_path = PATH + 'temp_test_photos/2_0_7344.png' # augmented 2
img_path = PATH + 'temp_test_photos/good/good.jpg' # license plate
img = np.array(Image.open(img_path))

def cut(img):
    # P x(30:160) y(150:245)
    P =  img[150:245, 30:160]
    P = cv2.resize(P,my_dim_rev)
    # ID x(165:285) y(150:245)
    ID = img[150:245, 165:285]
    ID = cv2.resize(ID,my_dim_rev)
    # A1 x(30:85) y(290:330) 
    A1 = img[290:330, 30:85]
    A1 = cv2.resize(A1,my_dim_rev)
    # A2 x(85:135) y(290:330)
    A2 = img[290:330, 85:135]
    A2 = cv2.resize(A2,my_dim_rev)
    # N1 x(180:230) y(280:325)
    N1 = img[280:325, 180:230]
    N1 = cv2.resize(N1,my_dim_rev)
    # N2 x(230:275) y(280:325)
    N2 = img[280:325, 230:275]
    N2 = cv2.resize(N2,my_dim_rev)
    return P
    # return P, ID, A1, A2, N1, N2

def arr_to_char(one_hot):
    val_index = np.argmax(one_hot)
    return my_str[val_index]

cv2.imshow('img',img)

img_to_pred = np.expand_dims(cut(img), axis=0)
cv2.imshow('to predict', img_to_pred[0])
y_predict = conv_model.predict(img_to_pred)[0]
val = arr_to_char(y_predict)
cv2.waitKey(0)
print(val)

