import yaml
import gym
import numpy as np
from argparse import Namespace
import time
from Overtaking.Controllers import PureRewardController as PRC
import SimulationBase
from Overtaking.Util.Map import Map

import torch

if __name__ == '__main__':

    f1map = Map('config_example_map_filled.yaml')

    speeds = [i for i in torch.arange(3.0, 3.01, .3)]
    angles = [i for i in torch.arange(-.3, .3001, .005)]
    # intstantiate controllers for agents
    controller = PRC.PureRewardController(f1map, speeds=speeds, angles=angles)

    SimulationBase.SimulateEgo(f1map, controller, np.array([0, 0, 0]))
