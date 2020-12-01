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
from keras import models


my_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
sylvia_path = '/home/sylvia'
cnn_path = '/ros_ws/src/my_controller/cnn_training/'
PATH = sylvia_path + cnn_path
model_path = PATH + 'alphanumeric_detector_model_v2'

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


class plate_decrypter:
    
    def __init__(self):
        self.conv_model = models.load_model(model_path, compile=True)
        self.bridge = CvBridge()
        self.license_value_pub = rospy.Publisher('/plate_value', String, queue_size=1)
        self.cropped_plate_sub = rospy.Subscriber("/cropped_plate", Image, self.callback)


    def prediction(self, img, P=False):
        '''Returns the CNN prediction and confidence value'''
        img = np.expand_dims(img, axis=0)
        y_predict = self.conv_model.predict(img)[0]
        index = np.argmax(y_predict)
        value = my_str[index]
        confidence = y_predict[index]
        if P:
            P_conf = y_predict[15]
            return value, P_conf
        return value, confidence


    def callback(self, data):
        try:
            cropped_plate = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cropped_plate = np.expand_dims()
        P, ID, A1, A2, N1, N2 = cut(cropped_plate)

        P_val, P_conf = self.prediction(P, P=True)

        if P_val == 'P' and P_conf > 0.9:
            ID_val, ID_conf = self.prediction(ID)
            A1_val, A1_conf = self.prediction(A1) #also check that these are letters...
            A2_val, A2_conf = self.prediction(A2)
            N1_val, N1_conf = self.prediction(N1) #... and these are numbers
            N2_val, N2_conf = self.prediction(N2)

            total_conf = (ID_conf+A1_conf+A2_conf+N1_conf+N2_conf)/5.0

            plate_str = str(ID_val) + str(A1_val) + str(A2_val) + str(N1_val) + str(N2_val)

            pub_str = plate_str + str(total_conf)
            
            ## PUBLISH PLATE
            ## message is in format: #AA##confidence 
            rospy.loginfo("License plate read: " + pub_str)
            self.license_value_pub.publish(pub_str)

            # cv2.imshow("License plate read by neural net", cropped_plate)
            # cv2.imshow('self last frame', self.last_frame)
            # cv2.waitKey(1)


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
