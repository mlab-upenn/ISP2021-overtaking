#!/usr/bin/env python

import rospy
import rospkg
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from math_util import geometric_operations as geo_ops
from math_util import geometric_types as geo_types
from math_util import occupancy_grid
from math_util import vector_operations as vec_ops
from math_util import trajectory_loader as traj_loader

# importTest()
class pure_pursuit():
    def __init__(self):
        #set up parameters
        self.intersection_radius_self = 1
        self.curve_p = .2
        self.traj = []


        rospy.init_node("pure_pursuit")
        self.drive_pub = rospy.Publisher("drive", AckermannDriveStamped,queue_size=1)
        self.path_pub = rospy.Publisher("path", Marker, queue_size=1)
        self.target_pub = rospy.Publisher
        self.intersection_pub = rospy.Publisher('intersections', Marker, queue_size=1)
        self.pose_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)
        self.occ_grid = occupancy_grid.StaticOccupancyGrid()

        self.traj = traj_loader.create_trajectory_from_file(rospkg.RosPack().get_path('pure_pursuit') + '/data/SampleTrajectory.csv')
        marker = Marker()
        marker.color.a = 1
        marker.color.r = 1
        marker.color.b = 0
        marker.color.g = 0
        marker.scale.x = .1

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.header.frame_id = "map"
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP

        for i in range(len(self.traj)+1):
            t = self.traj[i%len(self.traj)]
            pos = Point()
            pos.x = t.pt1[0]
            pos.y = t.pt1[1]
            marker.points.append(pos)

            print(t.pt1)


        self.path_pub.publish(marker)

        rospy.spin()
        pass

    def odom_callback(self, odom):
        if(len(self.traj) == 0):
            return
        #run pure pursuit controller and publish drive message
        msg = AckermannDriveStamped()
        msg.drive.speed = 1
        msg.drive.steering_angle = 0

        pose = odom.pose.pose

        vec_ops.get_forward_vector(pose)

        self_radius = geo_types.circle(np.array([pose.position.x,pose.position.y]), self.intersection_radius_self)

        intersections = self.get_trajectory_intersections(self_radius)

        world_target = self.get_forwardmost_intersection(pose, intersections)
        print('world target:', world_target)
        if(world_target is None):
            return
        car_target = vec_ops.world_to_car(pose,world_target)

        # print('car_target: ', car_target)
        #pure pursuit steering:
        msg.drive.steering_angle = self.curve_p*2*car_target[0]/np.linalg.norm(car_target)

        # print('steering angle: ', msg.drive.steering_angle)
        self.drive_pub.publish(msg)

        inter_marker = self.make_intersections_marker(intersections)
        # inter_marker = self.make_intersections_marker([(0, world_target)])
        self.intersection_pub.publish(inter_marker)



    def get_forwardmost_intersection(self,pose, intersections):
        if(len(intersections) == 0):
            return None
        forward = vec_ops.get_forward_vector(pose)
        position = np.array([pose.position.x, pose.position.y])
        best_inter = intersections[0][1]
        highest_forwardness = -np.infty
        print(forward)
        for (i,inter) in intersections:
            forwardness = np.dot(inter-position, forward.flatten())
            if(forwardness > highest_forwardness):
                highest_forwardness = forwardness
                best_inter = inter

        return best_inter


    def get_trajectory_intersections(self, circle):
        intersections = []
        for i,segment in enumerate(self.traj):
            inters = geo_ops.intersection_of_circle_and_segment(circle, segment)
            pairs = [(i, seg) for seg in inters]
            intersections.extend(pairs)
        return intersections


    def make_intersections_marker(self, intersections):
        marker = Marker()
        marker.color.a = 1
        marker.color.r = 0
        marker.color.b = 1
        marker.color.g = 0
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.header.frame_id = "map"
        marker.action = Marker.ADD
        marker.type = Marker.SPHERE_LIST
        for (i, inter) in intersections:
            pos = Point()
            pos.x = inter[0]
            pos.y = inter[1]
            marker.points.append(pos)
        return marker





if __name__ == '__main__':
    pp = pure_pursuit()