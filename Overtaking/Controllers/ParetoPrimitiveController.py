import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.TreeMotionPrimitives import TreeMotionPrimitive
from .PrimitiveBasedControllerSuper import PrimitiveBasedControllerSuper
from ..Util.ParetoFront import pareto_front


class ParetoPrimitiveController(PrimitiveBasedControllerSuper):

    def __init__(self, f1map, primitive_class, static_risk_factor=1, dynamic_risk_factor=0, reward_factor=0, dynamic_risk_thresh = .15, local_grid_world_size = 10, resolution=120):
        ## from renderer.py in f1tenth gym for loading map image
        self.MP = primitive_class
        self.static_risk_factor = static_risk_factor
        self.dynamic_risk_factor = dynamic_risk_factor
        self.reward_factor = reward_factor
        self.dynamic_risk_thresh = dynamic_risk_thresh
        super().__init__( f1map, None, None, local_grid_world_size, resolution)

    def initialize_primitives(self, speeds, angles):
        pass

    def plan(self, pose):
        local_obstacles = self.map.sample_obstacles(pose, self.local_grid_size, self.resolution)
        static_risks = self.get_risks(local_obstacles)
        static_risks /= np.linalg.norm(static_risks)
        dynamic_risks, _ = self.get_dynamic_risks(pose, self.dynamic_risk_thresh)
        dynamic_risks /= np.linalg.norm(dynamic_risks)
        rewards = self.get_rewards(self.map.sample_reward(pose, self.local_grid_size, self.resolution))
        rewards /= -np.linalg.norm(rewards)
        
        # 3D case 
        print(rewards)
        print(dynamic_risks)
        print(static_risks)

        pareto_optimal_set = pareto_front(np.stack((static_risks, dynamic_risks, rewards)).T)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        suboptimal_set = np.zeros(static_risks.shape[0], dtype = bool)
        suboptimal_set[pareto_optimal_set] = True
        suboptimal_set = np.where(~suboptimal_set)
        ax.scatter(static_risks[pareto_optimal_set], dynamic_risks[pareto_optimal_set], rewards[pareto_optimal_set], 'o')
        ax.scatter(static_risks[suboptimal_set], dynamic_risks[suboptimal_set], rewards[suboptimal_set], '^')
        ax.set_xlabel('Static Risks')
        ax.set_ylabel('Dynamic Risks')
        ax.set_zlabel('Rewards')
        plt.show()

        # 2D case
        # risks = static_risks + dynamic_risks
        # pareto_optimal_set = pareto_front(np.stack((risks, rewards)).T)
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # suboptimal_set = np.zeros(static_risks.shape[0], dtype = bool)
        # suboptimal_set[pareto_optimal_set] = True
        # suboptimal_set = np.where(~suboptimal_set)
        # ax.scatter(risks[pareto_optimal_set], rewards[pareto_optimal_set])
        # ax.scatter(risks[suboptimal_set], rewards[suboptimal_set])
        # ax.set_xlabel('Risks')
        # ax.set_ylabel('Rewards')
        # plt.show()
        
        control_choice = pareto_optimal_set[torch.argmin(dynamic_risks[pareto_optimal_set])]
        speed, angle = self.MP.get_control_for(control_choice)

        return speed, angle, self.MP.primitives[control_choice]


