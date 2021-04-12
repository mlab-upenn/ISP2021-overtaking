import yaml
import gym
import numpy as np
from argparse import Namespace
import time
from Overtaking.Controllers import PureRiskController as PRC
from Overtaking.Controllers import TreePrimitiveController as TPC
import SimulationBase
from Overtaking.Util.Map import Map

import torch

if __name__ == '__main__':

    f1map = Map('config_example_map_filled.yaml')

    speeds = [i for i in torch.arange(3.0, 3.01, 1.0)]
    angles = [i for i in torch.arange(-.2, .2001, .03)]
    # intstantiate controllers for agents
    controller = TPC.TreePrimitiveController(f1map, 2, speeds=speeds, angles=angles)

    SimulationBase.SimulateEgo(f1map, controller, np.array([0, 2, 0]))
