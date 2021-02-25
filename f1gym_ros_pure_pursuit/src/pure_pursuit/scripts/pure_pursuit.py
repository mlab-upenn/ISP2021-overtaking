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
    def __init__(self, speed = 3,name='pure_pursuit', drive_pub_name='drive',pose_sub = 'odom', opp_pose_sub='opp_odom', try_to_overtake=False, traj_path = '/data/SampleTrajectory.csv'):
        #set up parameters
        self.intersection_radius_self = 2
        self.curve_p = .4
        self.opp_pass_bubble = .6
        self.speed = speed
        self.try_to_overtake = try_to_overtake
        self.traj = []
        self.opp_pose = None
        self.ego_pose = None


        rospy.init_node(name)
        self.drive_pub = rospy.Publisher(drive_pub_name, AckermannDriveStamped,queue_size=1)
        self.path_pub = rospy.Publisher(name+"_path_array", MarkerArray, queue_size=1)
        self.target_pub = rospy.Publisher
        self.intersection_pub = rospy.Publisher(name+'_intersections', Marker, queue_size=1)
        self.pass_lane_pub = rospy.Publisher(name+'_pass_lanes', MarkerArray, queue_size=1)
        self.occ_grid = occupancy_grid.StaticOccupancyGrid()

        self.traj = traj_loader.create_trajectory_from_file(rospkg.RosPack().get_path('pure_pursuit') + traj_path)
        marker = self.make_traj_marker(self.traj, name='path')

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        
        self.path_pub.publish(marker_array)

        self.opp_pose_sub = rospy.Subscriber(opp_pose_sub, Odometry, self.opp_odom_callback)
        self.pose_sub = rospy.Subscriber(pose_sub, Odometry, self.odom_callback)
        rospy.Timer(rospy.Duration(.01), self.control_callback)


    def control_callback(self,event):
        if((self.ego_pose is None) | (not self.occ_grid.loaded)):
            return
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = 0

        pose = self.ego_pose

        self_radius = geo_types.circle(np.array([pose.position.x,pose.position.y]), self.intersection_radius_self)

        intersections = self.get_trajectory_intersections(self_radius, self.traj)

        world_target = self.get_forwardmost_intersection(pose, intersections)

        overtake_goal_world = None
        if(world_target is None):
            return
        car_target = vec_ops.world_to_car(pose, world_target)
        if(self.try_to_overtake):
            overtake_goal_world = self.overtake_check(pose)
            # print(overtake_goal_world)
            if(overtake_goal_world is not None):
                car_target = vec_ops.world_to_car(pose, overtake_goal_world)
                # print(car_target)
                pass
            else:
                # print('no overtake')
                pass


        msg.drive.steering_angle = self.curve_p*2*car_target[1]/np.linalg.norm(car_target)

        # print('steering angle: ', msg.drive.steering_angle)
        self.drive_pub.publish(msg)

        # inter_marker = self.make_intersections_marker(intersections)
        if(overtake_goal_world is not None):
            # rospy.loginfo('overtake target')
            inter_marker = self.make_intersections_marker([(0, overtake_goal_world)])
            self.intersection_pub.publish(inter_marker)

    def odom_callback(self, odom):
        self.ego_pose = odom.pose.pose


    def opp_odom_callback(self, pose):
        self.opp_pose = pose.pose.pose


    def overtake_check(self,pose):
        if(self.opp_pose is None):
            # print('opp pose none')
            return None
        opp_pose = np.array([self.opp_pose.position.x, self.opp_pose.position.y])

        ego_pose = np.array([pose.position.x, pose.position.y])

        # print(ego_pose)


        if(np.linalg.norm(opp_pose-ego_pose) > self.intersection_radius_self):
            # print('not close to opp')
            return None
        #first generate paths
        opp_left = vec_ops.get_left_vector(self.opp_pose)

        opp_forward = vec_ops.get_forward_vector(self.opp_pose)
        #world space passing points
        pass_points1 = [opp_left*p + opp_forward*(-.3) + opp_pose for p in np.linspace(-1, -self.opp_pass_bubble, 4)]
        pass_points1.extend([opp_left*p + opp_forward*(-.3) + opp_pose for p in np.linspace(self.opp_pass_bubble, 1, 4)])

        pass_points2 = [opp_left*p + opp_forward*(.3) + opp_pose for p in np.linspace(-1, -self.opp_pass_bubble, 4)]
        pass_points2.extend([opp_left*p + opp_forward*(.3) + opp_pose for p in np.linspace(self.opp_pass_bubble, 1, 4)])

        #find reconvergence point
        opp_circle = geo_types.circle(opp_pose, self.intersection_radius_self)
        opp_inters = self.get_trajectory_intersections(opp_circle, self.traj)
        reconvergence_point = self.get_forwardmost_intersection(self.opp_pose, opp_inters)

        if(np.linalg.norm(reconvergence_point-ego_pose) < self.intersection_radius_self):
            # print('close to reconvergence')
            return None

        #generate passing lines
        possible_pass = []

        for p1,p2 in zip(pass_points1,pass_points2):
            traj = [geo_types.line_segment(ego_pose, p1), geo_types.line_segment(p1, p2), geo_types.line_segment(p2, reconvergence_point)]
            add = True

            for t in traj:
                # print('checking line: ', str(t))
                if(self.occ_grid.checkForCollision(t)):
                    add = False
                    pass
            if(add):
                possible_pass.append((traj))


        #if no possible passes:
        if(len(possible_pass) == 0):
            # print('no possible paths')
            return None

        #otherwise find shortest path
        shortest_traj = None
        shortest_dist = np.infty

        for traj in possible_pass:
            d = 0
            for seg in traj:
                d += seg.length()
            if(d < shortest_dist):
                shortest_dist = d
                shortest_traj = traj


        marker_array = MarkerArray()
        for traj in possible_pass:
            marker = self.make_traj_marker(traj, id=len(marker_array.markers))

            marker_array.markers.append(marker)
        self.pass_lane_pub.publish(marker_array)


        #intersection of small radius with pass traj

        ego_circle_small = geo_types.circle(ego_pose, self.intersection_radius_self/2)
        pass_inter = self.get_trajectory_intersections(ego_circle_small, shortest_traj)
        targ = self.get_forwardmost_intersection(pose, pass_inter)
        return targ



    def get_forwardmost_intersection(self,pose, intersections):
        if(len(intersections) == 0):
            return None
        forward = vec_ops.get_forward_vector(pose)
        position = np.array([pose.position.x, pose.position.y])
        best_inter = intersections[0][1]
        highest_forwardness = -np.infty
        for (i,inter) in intersections:
            dir = inter-position
            forwardness = np.dot(dir/np.linalg.norm(dir), forward.flatten())
            if(forwardness > highest_forwardness):
                highest_forwardness = forwardness
                best_inter = inter

        return best_inter


    def get_trajectory_intersections(self, circle, traj):
        intersections = []
        for i,segment in enumerate(traj):
            inters = geo_ops.intersection_of_circle_and_segment(circle, segment)
            pairs = [(i, seg) for seg in inters]
            intersections.extend(pairs)
        return intersections

    def make_traj_marker(self, traj, id = 0, name = 'pass'):
        marker = Marker()
        marker.color.a = 1
        marker.color.r = 1
        marker.color.b = 0
        marker.color.g = 0
        marker.scale.x = .05

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.header.frame_id = "map"
        marker.ns = name
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.id = id
        marker.lifetime = rospy.Duration(.1)


        for i in range(len(traj)):
            t = traj[i % len(traj)]
            pos = Point()
            pos.x = t.pt1[0]
            pos.y = t.pt1[1]
            marker.points.append(pos)
        t = traj[len(traj)-1]
        pos = Point()
        pos.x = t.pt2[0]
        pos.y = t.pt2[1]
        marker.points.append(pos)
        return marker

    def make_intersections_marker(self, intersections):
        marker = Marker()
        marker.color.a = 1
        marker.color.r = 0
        marker.color.b = 1
        marker.color.g = 0
        marker.scale.x = .3
        marker.scale.y = .3
        marker.scale.z = .3

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.header.frame_id = "map"
        marker.action = Marker.ADD
        marker.type = Marker.SPHERE_LIST
        marker.lifetime = rospy.Duration(.1)

        for (i, inter) in intersections:
            pos = Point()
            pos.x = inter[0]
            pos.y = inter[1]
            marker.points.append(pos)
        return marker





# if __name__ == '__main__':
#     pp = pure_pursuit()
#
#     rospy.spin()