import gym
import time
import torch
import numpy as np
from Overtaking.Controllers import PureRiskController as PRC
from Overtaking.Controllers import DynamicObsController as DOC
import SimulationBase
from Overtaking.Util.Map import Map


if __name__ == '__main__':

    # load map
    f1map = Map('config_example_map_filled.yaml')

    speeds = [i for i in torch.arange(5.0, 5.01, .3)]
    angles = [i for i in torch.arange(-.3, .3001, .05)]
    # intstantiate controllers for agents
    controller = DOC.DynamicObstacleController(f1map,speeds, angles)
    opp_controller = PRC.PureRiskController(f1map, speeds, angles)

    SimulationBase.SimulateWithOpponent(f1map, controller, np.array([0, -4, 0]), opp_controller, np.zeros((3)))
