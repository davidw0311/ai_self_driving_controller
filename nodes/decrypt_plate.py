#!/usr/bin/env python

import rospy
import roslib
roslib.load_manifest('my_controller')
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool

import sys
import random

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
# from PIL import Image as Image_PIL
import tensorflow as tf
from keras import models
import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

david_path = '/home/davidw0311'
cnn_path = '/ros_ws/src/my_controller/cnn_training/'
PATH = david_path + cnn_path
model_path = PATH + 'detector_model_v7_8epoch_lenet'

my_dim_rev = (105, 150)

def cut(img):
    '''cuts the cropped license plate and returns a crop of each character'''
    P = cv2.resize(img[150:245, 30:160], my_dim_rev)
    ID = cv2.resize(img[150:245, 165:285], my_dim_rev)
    A1 = cv2.resize(img[280:330, 20:76], my_dim_rev)
    A2 = cv2.resize(img[280:330, 76:135], my_dim_rev)
    N1 = cv2.resize(img[280:330, 180:227], my_dim_rev)
    N2 = cv2.resize(img[280:330, 227:280], my_dim_rev)
    return P, ID, A1, A2, N1, N2

def arr_to_char(one_hot):
    val_index = np.argmax(one_hot)
    print('val index', val_index)
    return my_str[val_index]

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

class plate_decrypter:
    
    def __init__(self):
        print('here')
        self.sess = keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()
        self.conv_model = load_model(model_path, compile=True)
        print('loaded model')
        # print(self.conv_model.summary())
        
        self.bridge = CvBridge()
        self.license_value_pub = rospy.Publisher('/plate_value', String, queue_size=1)
        self.cropped_plate_sub = rospy.Subscriber("/cropped_plate", Image, self.callback)
    
    def predict(self, c):
        y_predict = conv_model.predict(np.array([c]))
        return y_predict

    def callback(self, data):
        try:
            cropped_plate = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        if cropped_plate is not None:
            uh = 123
            us = 255
            uv = 228
            lh = 107
            ls = 102
            lv = 79
            lower_hsv = np.array([lh,ls,lv])
            upper_hsv = np.array([uh,us,uv])
            
            plate_HSV = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2HSV)
            thresh_blue = cv2.inRange(plate_HSV, lower_hsv, upper_hsv)
            thresh_black = cv2.inRange(plate_HSV, (0,0,0), (0,0,67))
            
            # ret, thresh = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
            total_thresh = cv2.bitwise_or(thresh_blue, thresh_black)
            total_thresh = ~total_thresh
            # cv2.imshow('thresh', total_thresh)

            P, ID, A1, A2, N1, N2 = cut(total_thresh)
            
            # cv2.imshow('P', P)
            # cv2.imshow('ID', ID)
            # cv2.imshow('A1', A1)
            # cv2.imshow('A2', A2)
            # cv2.imshow('N1', N1)
            # cv2.imshow('N2', N2)
            # cv2.waitKey(1)

            with self.graph.as_default():
                set_session(self.sess)
                def get_prediction(a, isnum):
                    a = cv2.merge((a,a,a))
                    a_predictions = self.conv_model.predict(np.array([a]))[0]
                    if isnum:
                        a_index = np.argmax(a_predictions[:10])
                    else:
                        a_index = np.argmax(a_predictions[10:]) + 10
                    a_confidence = a_predictions[a_index]
                    a_prediction = decode_one_hot(a_index)
                    
                    return a_prediction, a_confidence, a_predictions
                
                P_prediction, P_confidence, P_predictions = get_prediction(P, isnum=False)
                # print(P_prediction,'P predictions', np.round(np.array(P_predictions), 3))
                ID_prediction, ID_confidence, ID_predictions = get_prediction(ID, isnum=True)
                # print(ID_prediction,'ID predictions', np.round(np.array(ID_predictions), 3))
                A1_prediction, A1_confidence, A1_predictions = get_prediction(A1, isnum=False)
                A2_prediction, A2_confidence, A2_predictions = get_prediction(A2, isnum=False)
                # print(A1_prediction,'a1 predictions', np.round(np.array(A1_predictions), 3))
                N1_prediction, N1_confidence, N1_predictions = get_prediction(N1, isnum=True)
                N2_prediction, N2_confidence, N2_predictions = get_prediction(N2, isnum=True)

            prediction = ID_prediction + A1_prediction + A2_prediction + N1_prediction + N2_prediction
            confidence = (ID_confidence + A1_confidence + A2_confidence + N1_confidence + N2_confidence)/5
            self.license_value_pub.publish(prediction+str(confidence))
            print(prediction, confidence)
        self.license_value_pub.publish('')


def main(args):
    rospy.init_node('plate_decrypter', anonymous=True)   
    pd = plate_decrypter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
