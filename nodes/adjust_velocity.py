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
from std_msgs.msg import Float32, Bool, String

WIDTH = 1280
HEIGHT = 720

# taken from https://github.com/ivmech/ivPID
class PID:

    def __init__(self, P=0.0, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

def nothing(x):
    pass

#pedestrial array length
pal = 15
damping_vel = Twist()
damping_vel.linear.x = 0.12
damping_vel.angular.z = 0

class velocity_control:

    def __init__(self):
        self.init_time = time.time()
        self.moving_pedestrian_array = []
        for i in range(pal):
            # stationary = False, moving = True
            self.moving_pedestrian_array.append(False)
        self.last_time_at_crosswalk = 0
        self.at_intersection = False
        self.stop_at_crosswalk = False
        self.centroid_location = 0
        self.controller_state = ''
        self.timer_started = False
        self.velocity_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
      
        self.centroid_location_sub = rospy.Subscriber('/centroid_location', Float32, self.callback)
        self.at_crosswalk_sub = rospy.Subscriber('/at_crosswalk', Bool, self.at_crosswalk_callback)
        self.moving_pedestrian = rospy.Subscriber('/moving_pedestrian', Bool, self.moving_pedestrian_callback)
        self.at_intersection_sub = rospy.Subscriber('/at_intersection', Bool, self.at_intersection_callback)
        self.controller_state_sub = rospy.Subscriber('/controller_state', String, self.controller_state_callback)
        self.pid_controller = PID(0,0,0,time.time())
        
    def get_velocity(self, centroid):

        # turning_factor = pid_control(centroid)
        turning_factor = 1
        driving_speed = cv2.getTrackbarPos('driving speed', "PID Controller") / 100.0
        turning_speed = cv2.getTrackbarPos('turning speed', "PID Controller") / 100.0
        
        self.pid_controller.setKp(cv2.getTrackbarPos('proportional', "PID Controller"))
        self.pid_controller.setKd(cv2.getTrackbarPos('derivative', "PID Controller"))
        self.pid_controller.setKi(cv2.getTrackbarPos('integral', "PID Controller"))
        
        error = centroid - WIDTH/2
        self.pid_controller.update(error, time.time())

        scaling_factor = cv2.getTrackbarPos('scale factor', "PID Controller")
        pid_factor = self.pid_controller.output / scaling_factor
        # print("pid factor ", pid_factor)
        velocity = Twist()

        velocity.linear.x = driving_speed
        velocity.angular.z = turning_speed*pid_factor
        
        return velocity
    

    def callback(self, data):
        self.centroid_location = data.data
        if time.time() - self.last_time_at_crosswalk > 15:
            for i in range(pal):
                self.moving_pedestrian_array[i] = False 
        # print('centroid', self.centroid_location)
        cv2.imshow("PID Controller", np.zeros((1,400,3), np.uint8))
        cv2.waitKey(1)

        cv2.createTrackbar('driving speed','PID Controller',0,100,nothing)   
        cv2.createTrackbar('turning speed','PID Controller',0,100,nothing)
        cv2.createTrackbar('proportional','PID Controller',0,100,nothing)
        cv2.createTrackbar('derivative','PID Controller',0,100,nothing)
        cv2.createTrackbar('integral','PID Controller',0,100,nothing)
        cv2.createTrackbar('scale factor','PID Controller',1000,5000,nothing)
        
        if (time.time() - self.init_time) < 2.0:
            # cv2.setTrackbarPos('driving speed', 'PID Controller', 18)
            # cv2.setTrackbarPos('turning speed', 'PID Controller', 36)
            cv2.setTrackbarPos('driving speed', 'PID Controller', 0)
            cv2.setTrackbarPos('turning speed', 'PID Controller', 0)
            cv2.setTrackbarPos('proportional', 'PID Controller', 14)
            cv2.setTrackbarPos('derivative', 'PID Controller', 7)
            cv2.setTrackbarPos('integral', 'PID Controller', 3)
            cv2.setTrackbarPos('scale factor', 'PID Controller', 1000)
        give_a_boost = False

        # print('moving pedestrian array', self.moving_pedestrian_array)
        # print('sum of first 3 elements', sum(self.moving_pedestrian_array[0:3]))
        if self.stop_at_crosswalk and sum(self.moving_pedestrian_array[0:3]) == 0 and sum(self.moving_pedestrian_array) >= pal - 4:
            self.stop_at_crosswalk = False
            give_a_boost = True
            # print('LETS GOOOOOO')
            # print('pedestrian just started moving!')

        # print('stop at crosswalk ', self.stop_at_crosswalk)
        if self.stop_at_crosswalk:
            # print('stopped at crosswalk')
            velocity = Twist()
            velocity.linear.x = 0
            velocity.angular.z = 0
            self.velocity_pub.publish(damping_vel)
        else:
            velocity = self.get_velocity(self.centroid_location)
        
        if self.at_intersection:
            self.velocity_pub.publish(damping_vel)
            self.velocity_pub.publish(damping_vel)
            velocity.linear.x = 0
            velocity.angular.z = -0.8
            if time.time() - self.init_time < 5:
                velocity.angular.z = 0.8
        
        if give_a_boost:
            velocity.linear.x = velocity.linear.x * 1.3
        
        if self.timer_started:
            # print("speed: " + str(velocity.linear.x) + "  turn: " + str(velocity.angular.z) + "\n")
            # print('current time', time.time())
            self.velocity_pub.publish(velocity)

    def at_crosswalk_callback(self, data):
        # print('at cross walk', data.data)
        # print('last time at crosswalk', self.last_time_at_crosswalk)
        if data.data == True:
            # print('last time', self.last_time_at_crosswalk)
            # print('current time', time.time())
            # print('\n')
            if time.time() - self.last_time_at_crosswalk > 15:
                self.stop_at_crosswalk = True
                self.last_time_at_crosswalk = time.time()
            

    def moving_pedestrian_callback(self, data):
        if self.stop_at_crosswalk:
            for i in range(pal-2, -1, -1):
                self.moving_pedestrian_array[i+1] = self.moving_pedestrian_array[i]
                
            self.moving_pedestrian_array[0] = data.data

    def at_intersection_callback(self, data):
        self.at_intersection = data.data
    
    def controller_state_callback(self, data):
        self.controller_state = data.data
        if (data.data == 'timer_started'):
            self.timer_started = True




def main(args):
    rospy.init_node('velocity_adjuster', anonymous = True)
    vc = velocity_control()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
    cv2.destroyAllWindows()
    stop_vel = Twist()
    stop_vel.linear.x = 0.0
    stop_vel.angular.z = 0.0
    stop_publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    stop_publisher.publish(stop_vel)

if __name__ == '__main__':
    print('started adjust_velocity')
    main(sys.argv)
