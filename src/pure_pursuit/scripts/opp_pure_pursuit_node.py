#!/usr/bin/env python


import rospy
import pure_pursuit

if __name__ == '__main__':

    opp = pure_pursuit.pure_pursuit(speed=2.5, name='opp_pure_pursuit', drive_pub_name='opp_drive', pose_sub='opp_odom', try_to_overtake=False)
    rospy.spin()