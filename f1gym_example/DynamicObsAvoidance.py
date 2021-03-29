import gym
import time
import torch
import numpy as np
from Overtaking.Controllers import PureRiskController as PRC
from Overtaking.Controllers import DynamicObsController as DOC
from Overtaking.Util.Map import Map


if __name__ == '__main__':
    # load map
    f1map = Map('config_example_map_filled.yaml')

    # initialize environment
    env = gym.make('f110_gym:f110-v0', 
                   map=f1map.path, 
                   map_ext=f1map.ext, 
                   num_agents=2)
    start_offset = np.array([0, -4, 0])
    ego_start = f1map.start_pose + start_offset
    opp_start = f1map.start_pose
    obs, step_reward, done, info = env.reset(np.array([ego_start, opp_start]))

    # intstantiate controllers for agents
    controller = DOC.DynamicObstacleController(f1map)
    opp_controller = PRC.PureRiskController(f1map)

    # simulation loop
    while not done:
        start = time.time()
        
        poses = torch.tensor([obs['poses_x'], 
                              obs['poses_y'], 
                              obs['poses_theta']]).T.double()
        opp_speed, opp_steer, opp_prim = opp_controller.plan(poses[1])

        controller.update_opponent_data(poses[1], opp_prim, opp_controller.MP.time_field/opp_speed, opp_speed, opp_steer, opp_controller.MP.local_grid_size)
        speed, steer, prim = controller.plan(poses[0])


        end = time.time()

        # print(end-start)

        obs, step_reward, done, info = env.step(np.array([[steer, speed], [opp_steer, opp_speed]]))
        # laptime += step_reward

        planners = [(controller.MP.x, controller.MP.y, prim), 
                    (opp_controller.MP.x, opp_controller.MP.y, opp_prim)]
        planners = [(controller.MP.x, controller.MP.y, prim)]
        env.render(mode='human_fast', planner_data=planners)
        # env.render(mode='human_fast')

    # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)