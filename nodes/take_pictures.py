#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('my_controller')
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def take_picture():


class take_pictures:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        
        #cv2.imshow("Image window", processed_image)
        #cv2.waitKey(1)

    #self.centroid_location_pub.publish(location)