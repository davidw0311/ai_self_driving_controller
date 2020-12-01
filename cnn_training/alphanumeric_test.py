

#img_path = PATH + 'characters_pictures/J/plate_JP69_0.png' # J
#img_path = PATH + 'characters_pictures/E/plate_EB49_0.png' # E
# img_path = PATH + 'temp_test_photos/2_0_7344.png' # augmented 2
# img_path = PATH + 'temp_test_photos/good/good.jpg' # license plate
# img = np.array(Image.open(img_path))

# def cut(img):
#     # P x(30:160) y(150:245)
#     P =  img[150:245, 30:160]
#     P = cv2.resize(P,my_dim_rev)
#     # ID x(165:285) y(150:245)
#     ID = img[150:245, 165:285]
#     ID = cv2.resize(ID,my_dim_rev)
#     # A1 x(20:76) y(280:330) 
#     A1 = img[280:330, 20:76]
#     A1 = cv2.resize(A1,my_dim_rev)
#     # A2 x(76:135) y(280:330)
#     A2 = img[280:330, 76:135]
#     A2 = cv2.resize(A2,my_dim_rev)
#     # N1 x(180:227) y(280:330)
#     N1 = img[280:330, 180:227]
#     N1 = cv2.resize(N1,my_dim_rev)
#     # N2 x(227:280) y(280:330)
#     N2 = img[280:330, 227:280]
#     N2 = cv2.resize(N2,my_dim_rev)
#     return P
#     # return P, ID, A1, A2, N1, N2

from keras import models
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

my_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PATH = '/home/davidw0311/ros_ws/src/my_controller/cnn_training/'
model_path = PATH + 'alphanumeric_detector_model_v2'

conv_model = models.load_model(model_path, compile=True)
my_dim_rev = (105, 150)
def arr_to_char(one_hot):
    val_index = np.argmax(one_hot)
    print('val index', val_index)
    return my_str[val_index]


P = cv2.imread('testp.jpg',1)
print(type(P))
y_predict = conv_model.predict(np.array([P]))
print(y_predict)
val = arr_to_char(y_predict)
print(val)