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

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

generate_images = False

grass_centroids = []
for t in range(20):
    grass_centroids.append(0)

def process_frame(frame, last_cX, last_frame):

    original_frame = np.copy(frame)
    height = frame.shape[0]
    width = frame.shape[1]
    dilation_kernel = np.ones((39, 39), np.uint8)

    road_frame = cv2.inRange(original_frame, (70, 70, 70), (90, 90, 90))
    road_frame[0:int(height/2),:] = np.zeros_like(road_frame[0:int(height/2), :])
    kernel = np.ones((9, 9), np.uint8)
    lines_frame = cv2.inRange(original_frame, (245, 245, 245), (255, 255, 255))

    # detect parked cars

#     car_frame_light_blue = cv2.inRange(original_frame[int(height/2):, :], (180,80,80), (220, 120, 120))
#     car_frame_dark_blue = cv2.inRange(original_frame[int(height/2):, :], (90,0,0), (130, 30, 30))
#     car_frame = cv2.bitwise_or(car_frame_dark_blue, car_frame_light_blue)

    # hsv_threshold = cv2.inRange(hsv, (0, 0, 0), (5, 100, 255))
    roi_offset = int(height/3)
    license_mask_light_gray = cv2.inRange(original_frame, (97, 97, 97), (110, 110, 110))
    license_mask_dark_gray = cv2.inRange(original_frame, (190, 190, 190), (210, 210, 210))
    license_mask_other_gray = cv2.inRange(original_frame, (110, 110, 110), (130, 130, 130))
    license_mask = cv2.bitwise_or(license_mask_light_gray, license_mask_dark_gray)
    license_mask = cv2.bitwise_or(license_mask, license_mask_other_gray)
    license_mask = cv2.erode(license_mask, np.ones((3, 3), np.uint8), iterations=1)
    license_mask = cv2.dilate(license_mask, np.ones((7, 7), np.uint8), iterations=1)
    # cv2.imshow('license_mask', license_mask)


    cropped_plate = None

    _, car_contours, _ = cv2.findContours(license_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(car_contours) > 0:
        max_area = 0
        car_cX, car_cY = 0, 0

        biggest_car_contour = car_contours[0]
        for c in car_contours:
            this_area = cv2.contourArea(c)
            if this_area > max_area:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if cY > roi_offset:
                    car_cX, car_cY = (cX, cY)
                    max_area = this_area
                    biggest_car_contour = c

        rect = cv2.minAreaRect(biggest_car_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        height_of_box = max(abs(box[0][1] - box[1][1]), abs(box[0][1] - box[2][1]))
        width_of_box = max(abs(box[0][0] - box[1][0]), abs(box[0][0] - box[2][0]))


        close_contours=[]
        for c in car_contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print(car_cX, car_cY)
            # cv2.putText(blank, str(cX)+' '+str(cY), (cX, cY),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if abs(cY - car_cY) < 1.5*height_of_box and abs(cX - car_cX) < 0.75*width_of_box and cY != car_cY:
                # print('found one')
                close_contours.append((c,cX, cY))

        closest_contour = None
        min_diff = height  
        for c, close_cX, close_cY in close_contours:
            if math.sqrt((car_cY - close_cY)**2 +  (car_cX - close_cX)**2) < min_diff:
                closest_contour = c
                min_diff = math.sqrt((car_cY - close_cY)**2 +  (car_cY - close_cY)**2) < min_diff
        
        if closest_contour is not None:
            rect = cv2.minAreaRect(closest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(blank, [biggest_car_contour], 0, (0,255,0), 3)
            combined_contour = np.concatenate((biggest_car_contour, closest_contour), axis = 0)
        
        else:
            #sometimes the detection detect the whole sign as one contour
            combined_contour = biggest_car_contour

        rect = cv2.minAreaRect(combined_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        (x1, y1), (x2, y2), (x3,y3), (x4,y4) = box
        top_left_x = min([x1,x2,x3,x4])
        top_left_y = min([y1,y2,y3,y4])
        bot_right_x = max([x1,x2,x3,x4])
        bot_right_y = max([y1,y2,y3,y4])
        
        cropped_plate = original_frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
        if cropped_plate.shape[0] > 10 and cropped_plate.shape[1] > 10:
            cropped_plate = cv2.resize(cropped_plate, (300,400), interpolation=cv2.INTER_AREA)
            
            letters_of_plate = cv2.inRange(cropped_plate[100:300,20:280], (-1,-1,-1),(10,10,10))
            
            M = cv2.moments(letters_of_plate)
            if (M['m00'] > 5):
                
                # cv2.imshow('letters of plate', letters_of_plate)
                cv2.imshow('license plate', cropped_plate)

                if generate_images:
                    random_name = get_random_string(10)
                    name = '/home/davidw0311/plate_images/' + random_name + '.jpg'
                    cv2.imwrite(name, cropped_plate)
                    print('wrote image to', name )
                cv2.waitKey(1)
            else:
                cropped_plate = None

        else:
            cropped_plate = None
    
    # detect the stop_walk
    red_frame = cv2.inRange(original_frame, (0, 0, 245), (10, 10, 255))
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
        mse = np.sum((original_frame.astype("float") -
                      last_frame.astype("float"))**2)
        mse /= float(original_frame.shape[0] * original_frame.shape[1])
      #     print('mean squared error', mse)
        if (mse > 30):
            moving_pedestrian = True
            cv2.putText(original_frame, 'MOVING PEDESTRIAN!', (400, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    if (red_cX > 100):
        go_fast = True
    else:
        go_fast = False
      #     print('mean squared error', mse)

    box_frame = cv2.inRange(original_frame, (90, 0, 0), (130, 30, 30))
    box_frame = cv2.dilate(box_frame, dilation_kernel, iterations=2)

#     cv2.imshow('box frame', box_frame)
#     cv2.imshow('expanded box', expanded_box_frame)
    lines_frame = cv2.bitwise_or(lines_frame, box_frame)
    # cv2.imshow('box frame', box_frame)
    lines_frame = cv2.dilate(lines_frame, kernel, iterations=2)
    lines_frame = cv2.morphologyEx(lines_frame, cv2.MORPH_CLOSE, kernel)
    lines_frame = cv2.GaussianBlur(lines_frame, (7, 7), 0)

    road_frame = road_frame - box_frame
    road_frame = cv2.dilate(road_frame, kernel, iterations=2)
    road_frame = cv2.morphologyEx(road_frame, cv2.MORPH_CLOSE, kernel)
    road_frame = cv2.GaussianBlur(road_frame, (7, 7), 0)

    grass_frame = cv2.inRange(original_frame, (55, 125, 15), (85, 155, 45))
    grass_frame = cv2.erode(grass_frame, kernel, iterations=1)
    grass_frame = cv2.dilate(grass_frame, kernel, iterations=3)
    _, grass_contours, _ = cv2.findContours(
        grass_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    grass_cX, grass_cY = width, height
    for c in grass_contours:
        if cv2.contourArea(c) > 8000:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if (abs(cX-width/2) < abs(grass_cX-width/2)):
                grass_cX, grass_cY = (cX, cY)
            cv2.circle(original_frame, (cX, cY), 10, (200, 100, 200), -1)
    cv2.circle(original_frame, (grass_cX, grass_cY), 20, (200, 100, 200), -1)
    # cv2.circle(original_frame, (car_cX, car_cY), 30, (100, 150, 200), -1)
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
        cv2.putText(original_frame, 'INTERSECTION!!', (400, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)


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
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 25
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, lines=np.array([]),
    #                     minLineLength=min_line_length, maxLineGap=max_line_gap)

    lines = cv2.HoughLines(edges, rho=1, theta=math.pi/180,
                           threshold=100, min_theta=math.pi/180, max_theta=math.pi)
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
            cv2.line(original_frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

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
                intersect_points.append((x, y))

            i += 1

    sum_x, sum_y = 0, 0
    for point in intersect_points:
        sum_x += point[0]
        sum_y += point[1]
        cv2.circle(original_frame, point, 10, [0, 255, 0], -1)
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

    cv2.circle(original_frame, (red_cX, red_cY), 10, [0, 0, 0], -1)
    cv2.circle(original_frame, (road_cX, int(height/2)), 30, (255, 255, 0), -1)
    cv2.circle(original_frame, (line_cX, line_cY), 20, (0, 255, 255), -1)

    return original_frame, int((line_cX+2*road_cX)/3), at_crosswalk, moving_pedestrian, at_intersection, cropped_plate, go_fast


class image_converter:

    def __init__(self):
        self.centroid_location_pub = rospy.Publisher("centroid_location", Float32, queue_size=1)
        self.at_crosswalk_pub = rospy.Publisher("at_crosswalk", Bool, queue_size=1)
        self.at_intersection_pub = rospy.Publisher("at_intersection", Bool, queue_size=1)
        self.go_fast_pub = rospy.Publisher("go_fast", Bool, queue_size = 1)
        self.moving_pedestrian_pub = rospy.Publisher("moving_pedestrian", Bool, queue_size=1)
        self.cropped_plate_pub = rospy.Publisher("cropped_plate", Image, queue_size=1)
        self.detecting_plate_pub = rospy.Publisher("detecting_plate", Bool, queue_size = 1)
        self.timer_started = False
        self.last_location = 540
        self.last_frame = np.zeros((720, 1280, 3))
        self.bridge = CvBridge()
        self.controller_state_sub = rospy.Subscriber('/controller_state', String, self.controller_state_callback)
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)
    
    def controller_state_callback(self, data):
        self.controller_state = data.data
        if (data.data == 'timer_started'):
            self.timer_started = True

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        processed_image, location, at_crosswalk, moving_pedestrian, at_intersection, cropped_plate, go_fast = process_frame(
            np.copy(cv_image), self.last_location, self.last_frame)
        self.last_location = location
        self.last_frame = cv_image
        width, height = processed_image.shape[1], processed_image.shape[0]
        processed_image_resized = cv2.resize(processed_image, (int(
            width/1.5), int(height/1.5)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image window", processed_image_resized)
        # cv2.imshow('self last frame', self.last_frame)
        cv2.waitKey(1)
        # print('process image timer started', self.timer_started)
        if self.timer_started:
            self.centroid_location_pub.publish(location)
        if cropped_plate is not None:
            self.cropped_plate_pub.publish(
                self.bridge.cv2_to_imgmsg(cropped_plate, 'bgr8'))
            self.detecting_plate_pub.publish(True)
        else:
            self.detecting_plate_pub.publish(False)
        self.at_crosswalk_pub.publish(at_crosswalk)
        self.moving_pedestrian_pub.publish(moving_pedestrian)
        self.at_intersection_pub.publish(at_intersection)
        
        self.go_fast_pub.publish(go_fast)
    
        


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
