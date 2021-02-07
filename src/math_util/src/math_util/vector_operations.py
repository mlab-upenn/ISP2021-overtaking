
from scipy.spatial.transform import Rotation as R
import numpy as np
import geometry_msgs.msg as geo_msg

def world_to_car(pose, world_frame):
    #pose should be a pose msg
    car_pt = np.array([pose.position.x, pose.position.y])
    car_rot = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return car_rot.inv()@(world_frame - car_pt)



def car_to_world(pose, car_frame):
    car_pt = np.array([pose.position.x, pose.position.y])
    car_rot = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return (car_rot@car_frame) + car_pt


