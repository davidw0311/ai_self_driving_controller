#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool
import cv2
import rospy
import sys
import roslib
roslib.load_manifest('my_controller')
import random
import string

import tensorflow as tf
from tensorflow import keras

class plate_decrypter:
    
    def __init__(self):
        self.bridge = CvBridge()
        self.cropped_plate_sub = rospy.Subscriber("/cropped_plate", Image, self.callback)

    def callback(self, data):
        try:
            cropped_plate = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("License plate read by neural net", cropped_plate)
        # cv2.imshow('self last frame', self.last_frame)
        cv2.waitKey(1)


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
