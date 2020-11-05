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

def process_frame(frame, last_cX):
    original_frame = frame
    height = frame.shape[0]
    width = frame.shape[1]

    lines_frame = cv2.inRange(original_frame, (245,245,245), (255,255,255))
    gray_frame = cv2.cvtColor(frame[height-500:height,:], cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
    # cv2.imshow("grey", grey_frame)
    ret, thresh = cv2.threshold(gray_frame, 80, 90, cv2.THRESH_BINARY_INV)
    road_image = cv2.inRange(frame, (80, 80, 80), (90, 90, 90))
    
    _, contours, _ = cv2.findContours(road_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    blank_frame = np.full_like(frame, 0)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2500:
            # black_frame = cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
            cv2.fillPoly(blank_frame, pts=[cnt], color=(0,255,0))
    # cv2.imshow("thresh", thresh)

    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
    else:
        cX = last_cX
    cY = height - 200

    cv2.circle(blank_frame,(cX,cY), 10, (0,0,255), -1)

    return lines_frame, cX

class image_converter:

  def __init__(self):
    self.centroid_location_pub = rospy.Publisher("centroid_location", Float32)
    self.last_location = 400
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    processed_image, location = process_frame(cv_image, self.last_location)
    self.last_location = location
    cv2.imshow("Image window", processed_image)
    cv2.waitKey(1)

    self.centroid_location_pub.publish(location)


def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)