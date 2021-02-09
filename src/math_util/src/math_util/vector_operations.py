
from tf.transformations import quaternion_matrix, translation_matrix, translation_from_matrix
import numpy as np
import geometry_msgs.msg as geo_msg

def importTest():
    print('import successful for vector_operations')

def world_to_car(pose, world_frame):
    pt = pt_to_trans_pt(world_frame)
    car_trans = create_transform_from_pose(pose)
    ret = np.matmul(np.linalg.inv(car_trans), pt).flatten()[0:2]
    return ret


def car_to_world(pose, car_frame):
    car_trans = create_transform_from_pose(pose)
    pt = pt_to_trans_pt(car_frame)
    return np.matmul(car_trans, pt).flatten()[0:2]

def get_forward_vector(pose):

    car_rot = quaternion_matrix([ pose.orientation.x, pose.orientation.y, pose.orientation.z,pose.orientation.w])
    init = translation_matrix([0,1,0])
    forward = translation_from_matrix(np.matmul(car_rot,init))
    return forward[0:2]

def create_transform_from_pose(pose):
    trans = quaternion_matrix([ pose.orientation.x, pose.orientation.y, pose.orientation.z,pose.orientation.w])
    trans[0,3] = pose.position.x
    trans[1,3] = pose.position.y
    trans[2,3] = pose.position.z
    return trans

def pt_to_trans_pt(pt):
    return np.array([[pt[0]],[pt[1]],[0],[1]])
