#!/usr/bin/env python

import pure_pursuit
import rospy
if __name__ == '__main__':
    ego = pure_pursuit.pure_pursuit(try_to_overtake=False)
    rospy.spin()