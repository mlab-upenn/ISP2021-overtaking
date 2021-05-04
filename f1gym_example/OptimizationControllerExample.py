import gym
import time
import torch
import numpy as np
from Overtaking.Controllers import PrimitiveController as PC
from Overtaking.Controllers import OptimizationController as OC
from Overtaking.MotionPrimitives import TreeMotionPrimitives as TMP
from Overtaking.MotionPrimitives import MotionPrimitives as MP
import SimulationBase
from Overtaking.Util.Map import Map


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.autograd.set_detect_anomaly(True)

    # load map
    f1map = Map('config_example_map_filled.yaml')

    speeds = [i for i in torch.arange(3.7, 3.71, .3)]
    angles = [i for i in torch.arange(-.3, .3001, .02)]
    # instantiate controllers for agents

    resolution = 50
    size = 5


    opp_motion_primitives = MP.MotionPrimitive(speeds, angles, L=.33, p=.2, t_la=2.5, k1=.2, k2=.3, k3=.1, m=.1, c=.12, resolution=(resolution, resolution), local_grid_size=size)

    opp_controller = PC.PrimitiveController(f1map,opp_motion_primitives, resolution=resolution, local_grid_world_size=size)

    # speeds = [i for i in torch.arange(3.5, 4.01, .5)]
    # angles = [i for i in torch.arange(-.25, .251, .05)]
    #
    # depth = 3
    resolution = 20
    size = 5

    num_primitives = 50
    time_steps = 4
    speeds = torch.ones((num_primitives,time_steps))*4

    steering_angles = torch.range(-.2, .2, .4/(num_primitives-1)).unsqueeze(1).repeat((1,time_steps))
    steering_angles[:,(time_steps//2):] *= -1

    print('beginning optimization controller setup')

    controller = OC.OptimizationController(f1map, speeds, steering_angles, static_risk_factor=0, dynamic_risk_factor= .3, reward_factor=1, dynamic_risk_thresh=0.2, local_grid_world_size=size, resolution=resolution)
    print('finished optimization controller setup')

    SimulationBase.SimulateWithOpponent(f1map, controller, np.array([0, 2, 0]), opp_controller, np.array([0,6,0]))
