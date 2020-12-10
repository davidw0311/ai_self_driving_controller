#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Bool

import sys
import numpy as np
from matplotlib import pyplot as plt

team_ID = 'donuts'
team_psd = 'enph353'
comp_time = 4*60                  # should be 4*60 seconds

class reporter:
    
    def __init__(self):
        self.pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.state_pub = rospy.Publisher('/controller_state', String, queue_size=1)
        self.confirmed_time = rospy.get_time()
        self.plates = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }

        self.start_timer_str = self.msg(0,'strt')
        self.stop_timer_str = self.msg(-1,'stop')

        rospy.sleep(2)
        rospy.loginfo(self.start_timer_str)
        self.pub.publish(self.start_timer_str)
        self.state_pub.publish('timer_started')
        self.start_time = rospy.get_time()

        self.plate_val_sub = rospy.Subscriber('/plate_value', String, self.callback)


    def check_end(self, time):
        if (time - self.start_time) >= comp_time:
            rospy.loginfo(self.stop_timer_str)
            self.pub.publish(self.stop_timer_str)
        if not (self.plates[i] == 0 for i in list(self.plates)):
            rospy.loginfo(self.stop_timer_str)
            self.pub.publish(self.stop_timer_str)
        
    
    def msg(self, lp_loc, lp_id):
        ret_str = team_ID + ',' + team_psd + ',' + str(lp_loc) + ',' + str(lp_id)
        return ret_str
    

    def callback(self, data):
        ## Format: #AA##confidence
        data_str = data.data
        if data_str != '':
            loc = int(data_str[0])
            plate_id = data_str[1:5]
            conf = float(data_str[5:])

            # if loc == 7 and self.plates[1]==0: # this plate is probably at location 1
            #     loc = 1
            
            if loc < 1 or loc > 8:
                return

            if conf > self.plates[loc]:
                self.plates[loc] = conf
                license_plate_str = self.msg(loc, plate_id)   # replace loc and id
                # rospy.loginfo(license_plate_str)
                self.pub.publish(license_plate_str)
            
            all_outside_confirmed = True
            print('plate read states')
            for i in range(1,7):
                print(self.plates[i])
                if self.plates[i] < 0.965:
                    all_outside_confirmed = False
    
            print('all outside confirmed', all_outside_confirmed, '\n')


            if all_outside_confirmed:
                if self.confirmed_time is None:
                    self.confirmed_time = rospy.get_time()

            if all_outside_confirmed:
                if rospy.get_time() - self.confirmed_time > 20:
                    rospy.loginfo(self.stop_timer_str)
                    self.pub.publish(self.stop_timer_str)
                    print('stopped timer at', rospy.get_time())
                    # self.state_pub.publish('go_inside')
            # else:
            #     self.state_pub.publish('stay_outside')
                
            print(str(loc)+plate_id)
            
        self.check_end(rospy.get_time())


def main(args):
    rospy.init_node('reporter', anonymous=True)
    rep = reporter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)