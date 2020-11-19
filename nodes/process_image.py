#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import roslib
roslib.load_manifest('my_controller')
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math

grass_centroids = []
for t in range(20):
      grass_centroids.append(0)

def process_frame(frame, last_cX, last_frame):
    
    original_frame = frame
    height = frame.shape[0]
    width = frame.shape[1]

    road_frame = cv2.inRange(original_frame, (70,70,70),(90,90,90))
    road_frame[0:int(height/2),:] = np.zeros_like(road_frame[0:int(height/2),:])
    kernel = np.ones((9,9),np.uint8)
    lines_frame = cv2.inRange(original_frame, (245,245,245), (255,255,255))

    # detect the stop_walk
    red_frame = cv2.inRange(original_frame, (0,0,245),(10,10,255))
    M = cv2.moments(red_frame)
    if M["m00"] != 0:
        red_cX = int(M["m10"] / M["m00"])
        red_cY = int(M["m01"] / M["m00"])
    else:
        red_cX = 0
        red_cY = 0
    
    # cv2.imshow('red bar frame', red_frame)
    at_crosswalk = False
    moving_pedestrian = False
    if (red_cY > 600):
      #     cv2.imshow('original',original_frame)
      #     cv2.imshow('last frame', last_frame)
      #     cv2.waitKey(0)
          at_crosswalk = True
    if (red_cY > 100):
          mse = np.sum((original_frame.astype("float") - last_frame.astype("float"))**2)
          mse /= float(original_frame.shape[0] * original_frame.shape[1])
          if (mse > 40):
                moving_pedestrian = True
                cv2.putText(original_frame,'MOVING PEDESTRIAN!',(400,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    
      #     print('mean squared error', mse)
          

    dilation_kernel = np.ones((39,39), np.uint8)
    box_frame = cv2.inRange(original_frame, (90,0,0),(130,30,30))
    box_frame = cv2.dilate(box_frame, dilation_kernel, iterations = 1)
    
#     cv2.imshow('box frame', box_frame)
#     cv2.imshow('expanded box', expanded_box_frame)
    lines_frame = cv2.bitwise_or(lines_frame, box_frame)
    # cv2.imshow('box frame', box_frame)
    lines_frame = cv2.dilate(lines_frame, kernel, iterations = 2)
    lines_frame = cv2.morphologyEx(lines_frame, cv2.MORPH_CLOSE, kernel)
    lines_frame = cv2.GaussianBlur(lines_frame, (7,7), 0)
    
    road_frame = road_frame - box_frame
    road_frame = cv2.dilate(road_frame, kernel, iterations = 2)
    road_frame = cv2.morphologyEx(road_frame, cv2.MORPH_CLOSE, kernel)
    road_frame = cv2.GaussianBlur(road_frame, (7,7), 0)

    grass_frame = cv2.inRange(original_frame, (55,125,15),(85,155,45))
    grass_frame = cv2.erode(grass_frame, kernel, iterations=1)
    grass_frame = cv2.dilate(grass_frame, kernel, iterations=3)
    _, grass_contours, _ = cv2.findContours(grass_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    grass_cX, grass_cY = width, height
    for c in grass_contours:
          if cv2.contourArea(c) > 1500:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if (abs(cX-width/2) < abs(grass_cX-width/2)):
                      grass_cX, grass_cY = (cX, cY)
                cv2.circle(original_frame, (cX, cY), 10, (200, 100, 200), -1)
    cv2.circle(original_frame, (grass_cX, grass_cY), 20, (200, 100, 200), -1)

    if abs(grass_cX - width/2) < 200 and grass_cY > height/2 and np.sum(box_frame/255) < 80000:
          intersection_value = 1
    else:
          intersection_value = 0      
      
    for i in range(len(grass_centroids) - 2, -1, -1):
          grass_centroids[i+1] = grass_centroids[i]
    grass_centroids[0] = intersection_value
    
    at_intersection = False
    if sum(grass_centroids) > 15:
          at_intersection = True
          cv2.putText(original_frame,'INTERSECTION!!',(400,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    
    
#     cv2.putText(original_frame,str(grass_cX) +' '+ str(grass_cY),(400,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)

#     cv2.imshow('grass frame', grass_frame)
#     cv2.imshow('road_frame', road_frame)
    cv2.waitKey(1)
    low_thresh, high_thresh = 50, 150
    edges = cv2.Canny(road_frame, low_thresh, high_thresh)
    # edges = cv2.dilate(edges, kernel, iterations=1)

#     cv2.imshow('lines frame dilated', lines_frame)
#     cv2.imshow('edges', edges)
#     cv2.imshow('road frame', road_frame)
    cv2.waitKey(1)


    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, lines=np.array([]),
    #                     minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    lines = cv2.HoughLines(edges, rho=1, theta=math.pi/180, threshold=100, min_theta=math.pi/180, max_theta=math.pi)
    if lines is not None:
          i = 0
          while i < len(lines):
                line = lines[i]
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                j = i + 1
                while j < len(lines):
                      other_line = lines[j]
                      other_rho = lines[j][0][0]
                      other_theta = lines[j][0][1]
                      if math.atan(abs(-math.cos(theta)/math.sin(theta) + math.cos(other_theta)/math.sin(other_theta))) < math.pi/180 * 10: 
                      # abs(other_rho - rho) < 60 and abs(other_theta-theta) < math.pi/180 * 15:
                            lines = np.delete(lines, j, 0)
                            lines[i][0][0] = (other_rho + rho)/2
                            lines[i][0][1] = (other_theta + theta)/2
                      else:
                            j += 1
                i += 1
                      
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            # print('rho', rho, 'theta', theta)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            const = 2000
            pt1 = (int(x0 + const*(-b)), int(y0 + const*(a)))
            pt2 = (int(x0 - const*(-b)), int(y0 - const*(a)))
            cv2.line(original_frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    intersect_points = []
    slopes = []
    if lines is not None:
          i = 0
          while (i < len(lines)):
                line = lines[i]
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                m = -math.cos(theta)/math.sin(theta)
                slopes.append(m)
                b = rho/math.sin(theta)
                for next_line in lines[i+1:]:
                      next_rho = next_line[0][0]
                      next_theta = next_line[0][1]
                      next_m = -math.cos(next_theta)/math.sin(next_theta)
                      next_b = next_rho/math.sin(next_theta)

                      x = int((b - next_b)/(next_m - m))
                      y = int(m*x + b)
                      intersect_points.append((x,y))
                
                i += 1

    sum_x, sum_y = 0,0
    for point in intersect_points:
          sum_x += point[0]
          sum_y += point[1]
          cv2.circle(original_frame, point, 10, [0,255,0], -1)         
    if len(intersect_points) > 0:
          line_cX = int(sum_x / len(intersect_points))
          line_cY = int(sum_y / len(intersect_points))
    else:
          line_cX = last_cX
          line_cY = int(height/2)

    M = cv2.moments(road_frame)
    if M["m00"] != 0:
        road_cX = int(M["m10"] / M["m00"])
    else:
        road_cX = int(width/2)
    

    cv2.circle(original_frame, (red_cX, red_cY), 10, [0,0,0], -1)
    cv2.circle(original_frame, (road_cX, int(height/2)), 30, (255,255,0), -1)
    cv2.circle(original_frame,(line_cX,line_cY), 20, (0,255,255), -1)

    return original_frame, int((line_cX+2*road_cX)/3), at_crosswalk, moving_pedestrian, at_intersection

class image_converter:

  def __init__(self):
    self.centroid_location_pub = rospy.Publisher("centroid_location", Float32, queue_size = 1)
    self.at_crosswalk_pub = rospy.Publisher("at_crosswalk", Bool, queue_size = 1)
    self.at_intersection_pub = rospy.Publisher("at_intersection", Bool, queue_size = 1)
    self.moving_pedestrian_pub = rospy.Publisher("moving_pedestrian", Bool, queue_size = 1)
    self.last_location = 540
    self.last_frame = np.zeros((720,1280,3))
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    processed_image, location, at_crosswalk, moving_pedestrian, at_intersection = process_frame(np.copy(cv_image), self.last_location, self.last_frame)
    self.last_location = location
    self.last_frame = cv_image
    width, height = processed_image.shape[1], processed_image.shape[0]
    processed_image_resized = cv2.resize(processed_image, (int(width/2), int(height/2)) , interpolation = cv2.INTER_AREA)
    cv2.imshow("Image window", processed_image_resized)
    # cv2.imshow('self last frame', self.last_frame)
    cv2.waitKey(1)

    self.centroid_location_pub.publish(location)
    self.at_crosswalk_pub.publish(at_crosswalk)
    self.moving_pedestrian_pub.publish(moving_pedestrian)
    self.at_intersection_pub.publish(at_intersection)


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