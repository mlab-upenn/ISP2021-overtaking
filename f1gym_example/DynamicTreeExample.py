import gym
import time
import torch
import numpy as np
from Overtaking.Controllers import PureRiskController as PRC
from Overtaking.Controllers import DynamicObsController as DOC
from Overtaking.Controllers import TreePrimitiveController as TPC
import SimulationBase
from Overtaking.Util.Map import Map


if __name__ == '__main__':

    # load map
    f1map = Map('config_example_map_filled.yaml')

    speeds = [i for i in torch.arange(3.7, 3.71, .3)]
    angles = [i for i in torch.arange(-.3, .3001, .05)]
    # intstantiate controllers for agents
    opp_controller = PRC.PureRiskController(f1map,speeds, angles)

    speeds = [i for i in torch.arange(4.0, 4.01, .5)]
    angles = [i for i in torch.arange(-.2, .2001, .1)]
    controller = TPC.TreePrimitiveController(f1map, 2, speeds, angles)

    SimulationBase.SimulateWithOpponent(f1map, controller, np.array([0, 2, 0]), opp_controller, np.array([0,6,0]))
