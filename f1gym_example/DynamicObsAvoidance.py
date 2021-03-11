import yaml
import gym
import numpy as np
from argparse import Namespace
import time
from Overtaking.Controllers import PureRiskController as PRC
from Overtaking.Controllers import DynamicObsController as DOC
import torch


if __name__ == '__main__':

    # work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    with open('config_example_map_filled.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy - 4, conf.stheta],[conf.sx, conf.sy , conf.stheta]]))
    # env.render()


    controller = DOC.DynamicObstacleController(conf.map_path, conf.map_ext)

    opp_controller = PRC.PureRiskController(conf.map_path, conf.map_ext)

    while not done:
        start = time.time()
        opp_speed, opp_steer, opp_prim = opp_controller.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1])

        controller.update_opponent_data(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1], opp_prim, opp_controller.MP.time_field/opp_speed, opp_speed, opp_steer, opp_controller.MP.local_grid_size)
        speed, steer, prim = controller.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])


        end = time.time()

        # print(end-start)

        obs, step_reward, done, info = env.step(np.array([[steer, speed], [opp_steer, opp_speed]]))
        # laptime += step_reward

        env.render(mode='human_fast', planner_data = (controller.MP.x, controller.MP.y, prim) )
        # env.render(mode='human_fast')

    # print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)