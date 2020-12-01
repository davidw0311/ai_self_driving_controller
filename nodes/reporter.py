#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import numpy as np
from matplotlib import pyplot as plt

team_ID = 'donuts'
team_psd = 'enph353'
comp_time = 4*60                  # should be 4*60 seconds

class reporter:
    
    def __init__(self):
        self.pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.start_pub = rospy.Publisher('/controller_state', String, queue_size=1)

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

        rospy.sleep(1)
        rospy.loginfo(start_timer_str)
        pub.publish(start_timer_str)
        start_pub.publish('timer_started')
        self.start_time = rospy.get_time()

        self.plate_val_sub = rospy.Subscriber('/plate_value', String, callback)


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
        loc = int(data_str[0])
        plate_id = data_str[1:5]
        conf = float(data_str[5:])

        if loc == 7 and plates[1]==0: # this plate is probably at location 1
            loc = 1

        if conf > plates[loc]:
            license_plate_str = msg(loc, plate_id)   # replace loc and id
            rospy.loginf(license_plate_str)
            self.pub.publish(license_plate_str)
        
        self.check_end(rospy.get_time())


def main(args):
    rospy.init_node('reporter', anonymous=True)
    rep = reporter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)