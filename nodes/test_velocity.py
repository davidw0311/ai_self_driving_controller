#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import cv2 
import argparse
import roslib
import rospy
import sys
import time
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool, String

WIDTH = 1280
HEIGHT = 720

class velocity_control:
    
    def __init__(self):
        self.centroid_location_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.count = 0
    
    def callback(self, data):
        self.centroid_location = data.data
        velocity = Twist()
        rate = rospy.Rate(10)
        speed = 40

        velocity.linear.x = 0.01 * speed
        velocity.angular.z = 0
        self.velocity_pub.publish(velocity)
        rate.sleep()
        if self.count % 10 == 0:
            velocity.linear.x = 0.005 * speed
            self.velocity_pub.publish(velocity)
            rate.sleep()
            velocity.linear.x = 0
            velocity.linear.z = 0
            self.velocity_pub.publish(velocity)
        
        self.count += 1
    

def main(args):
    rospy.init_node('velocity_tester', anonymous = True)
    vc = velocity_control()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
       

if __name__ == '__main__':
    
    main(sys.argv)


    
    

