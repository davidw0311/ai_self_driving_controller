#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import roslib
roslib.load_manifest('my_controller')
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math

def process_frame(frame, last_cX):
    original_frame = frame
    height = frame.shape[0]
    width = frame.shape[1]

    road_frame = cv2.inRange(original_frame, (70,70,70),(90,90,90))
    road_frame[0:int(height/2),:] = np.zeros_like(road_frame[0:int(height/2),:])
    kernel = np.ones((9,9),np.uint8)
    lines_frame = cv2.inRange(original_frame, (245,245,245), (255,255,255))
    box_frame = cv2.inRange(original_frame, (110,10,10),(130,30,30))
    lines_frame = cv2.bitwise_or(lines_frame, box_frame)
    # cv2.imshow('box frame', box_frame)
    lines_frame = cv2.dilate(lines_frame, kernel, iterations = 2)
    lines_frame = cv2.morphologyEx(lines_frame, cv2.MORPH_CLOSE, kernel)
    lines_frame = cv2.GaussianBlur(lines_frame, (7,7), 0)
    
    road_frame = cv2.dilate(road_frame, kernel, iterations = 2)
    road_frame = cv2.morphologyEx(road_frame, cv2.MORPH_CLOSE, kernel)
    road_frame = cv2.GaussianBlur(road_frame, (7,7), 0)
    
    low_thresh, high_thresh = 50, 150
    edges = cv2.Canny(road_frame, low_thresh, high_thresh)
    # edges = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow('lines frame dilated', lines_frame)
    cv2.imshow('edges', edges)
    cv2.imshow('road frame', road_frame)
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
    print('slopes', slopes, 'total', sum(slopes))
    print('\n')
    sum_x, sum_y = 0,0
    for point in intersect_points:
          sum_x += point[0]
          sum_y += point[1]
          cv2.circle(original_frame, point, 10, [0,255,0], -1)         
    if len(intersect_points) > 0:
          cX = int(sum_x / len(intersect_points))
          cY = int(sum_y / len(intersect_points))
    else:
          cX = last_cX
          cY = int(height/2)
  
    # print('length of lines', len(lines))
    # while i < len(lines):
    #       print('i', i)
    #       line = lines[i]
    #       try:
    #         m = (line[0][3] - line[0][2]) / (line[0][1] - line[0][0]) #slope from x1, x2, y1, y2
    #         b = line[0][2] - m * line[0][0]
    #       except ZeroDivisionError as e:
    #         m = float('inf')
    #         b = None
    #       j = i + 1
    #       while j < len(lines):
    #             # print('j', j)
    #             # print('length of lines: ', len(lines))
    #             other_line = lines[j]
    #             # print('line', line)
    #             # print('other line', other_line)
    #             try:
    #               other_m = (other_line[0][3] - other_line[0][2]) / (other_line[0][1] - other_line[0][0]) #slope
    #               other_b = other_line[0][2] - other_m * other_line[0][0]
    #             except ZeroDivisionError as e:
    #               other_m = float('inf')
    #               other_b = None
    #             # print('m', m, 'other m', other_m)
    #             # print('difference in angle deg:',(abs(math.atan(m) - math.atan(other_m))) * 180/np.pi )
    #             if (abs(math.atan(m) - math.atan(other_m)))*180/math.pi < 10 and (abs(b - other_b) < 50):
    #                 lines = np.delete(lines, j, 0)
    #                 new_m = math.atan((math.tan(m) - math.tan(other_m))/2)
    #                 new_b = (b + other_b) / 2
    #                 left_y = new_b
    #                 right_y = (new_m * width + new_b)
    #                 top_x = int(-new_b/new_m)
    #                 bottom_x = (height - new_b)/m
    #                 if (left_y > 0 and left_y < height):
    #                       print('case A')
    #                       y1 = left_y
    #                       x1 = 0
        
    #                       if (top_x > 0 and top_x < width):
    #                             y2 = 0
    #                             x2 = top_x
    #                       elif (bottom_x > 0 and bottom_x < width):
    #                             y2 = height
    #                             x2 = bottom_x
    #                       else:
    #                             y2 = right_y
    #                             x2 = width
    #                 elif (top_x > 0 and top_x < width):
    #                       print('case B')
    #                       y1 = 0
    #                       x1 = top_x

    #                       if (bottom_x > 0 and bottom_x < width):
    #                             y2 = height
    #                             x2 = bottom_x
    #                       else:
    #                             y2 = right_y
    #                             x2 = width
    #                 else:
    #                       print('case C')
    #                       y1 = right_y
    #                       x1 = width
    #                       y2 = height
    #                       x2 = bottom_x
                    
    #                 lines[i] = [x1,x2,y1,y2]  
    #                 print('new line', lines[i])
    #               # print('before', lines.shape)
    #               # print('deleted line', lines[j])
                          
    #                   # print('after', lines.shape)
    #             else:
    #                   j += 1
    #             # print('after deleting', lines.shape)
    #       i += 1
    #       # print('here')

#trying to find the right most line
# edge_y = m*1280 - m*x1 + y1
# if edge_y < edge_y_min:
#   edge_y_min = edge_y
#   max_x1, max_x2, max_y1, max_y2 = x1, x2, y1, y2
# cv2.line(original_frame,(x1,y1),(x2,y2),(255,0,0),4)
    # max_x = 0
    # for line in lines:
    #       for x1,y1,x2,y2 in line:
    #             if max(x1, x2) > max_x:
    #                   max_x = max(x1,x2)
    #                   max_line = line
    #             cv2.line(original_frame,(x1,y1+int(height/2)),(x2,y2+int(height/2)),(255,0,0),4)
    # # cv2.line(original_frame, (max_x1,max_y1),(max_x2,max_y2), (0,0,0), 7)
    # for x1,y1,x2,y2 in max_line:
    #       cv2.line(original_frame,(x1,y1+int(height/2)),(x2,y2+int(height/2)),(0,255,0),7)
          
    # gray_frame = cv2.cvtColor(frame[height-500:height,:], cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
    # # cv2.imshow("grey", grey_frame)
    # ret, thresh = cv2.threshold(gray_frame, 80, 90, cv2.THRESH_BINARY_INV)
    # road_image = cv2.inRange(frame, (80, 80, 80), (90, 90, 90))
    
    # _, contours, _ = cv2.findContours(road_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # blank_frame = np.full_like(frame, 0)
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 2500:
    #         # black_frame = cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)
    #         cv2.fillPoly(blank_frame, pts=[cnt], color=(0,255,0))
    # cv2.imshow("thresh", thresh)

    M = cv2.moments(road_frame)
    if M["m00"] != 0:
        road_cX = int(M["m10"] / M["m00"])
    else:
        road_cX = int(width/2 -100)
    cv2.circle(original_frame, (road_cX, int(height/2)), 30, (255,255,0), -1)
    cv2.circle(original_frame,(cX,cY), 20, (0,255,255), -1)

    return original_frame, int((cX+2*road_cX)/3)

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