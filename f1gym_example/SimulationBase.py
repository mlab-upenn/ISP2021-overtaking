import gym
import time
import torch
import numpy as np



def SimulateWithOpponent(f1map, ego_controller, ego_start_offset, opp_controller, opp_start_offset):

    # initialize environment
    env = gym.make('f110_gym:f110-v0',
                   map=f1map.path,
                   map_ext=f1map.ext,
                   num_agents=2)
    ego_start = f1map.start_pose + ego_start_offset
    opp_start = f1map.start_pose + opp_start_offset
    obs, step_reward, done, info = env.reset(np.array([ego_start, opp_start]))
    while not done:
        start = time.time()

        poses = torch.tensor([obs['poses_x'],
                              obs['poses_y'],
                              obs['poses_theta']]).T.double()
        opp_speed, opp_steer, opp_prim = opp_controller.plan(poses[1])

        ego_controller.update_opponent_data(poses[1], opp_prim, opp_controller.MP.time_field / opp_speed, opp_speed,
                                        opp_steer, opp_controller.MP.local_grid_size)
        speed, steer, prim = ego_controller.plan(poses[0])
        print(speed)
        end = time.time()

        # print(end-start)

        obs, step_reward, done, info = env.step(np.array([[steer.detach().cpu(), speed.detach().cpu()], [opp_steer.detach().cpu(), opp_speed.detach().cpu()]]))
        # laptime += step_reward

        planners = [(ego_controller.MP.x.detach().cpu(), ego_controller.MP.y.detach().cpu(), prim.detach().cpu()),
                    (opp_controller.MP.x.detach().cpu(), opp_controller.MP.y.detach().cpu(), opp_prim.detach().cpu())]

        # planners = [(ego_controller.MP.x, ego_controller.MP.y, prim)]
        env.render(mode='human_fast', planner_data=planners)


def SimulateEgo(f1map, ego_controller, ego_start_offset):

    # initialize environment
    env = gym.make('f110_gym:f110-v0',
                   map=f1map.path,
                   map_ext=f1map.ext,
                   num_agents=1)
    ego_start = f1map.start_pose + ego_start_offset
    obs, step_reward, done, info = env.reset(np.array([ego_start]))
    while not done:
        start = time.time()

        poses = torch.tensor([obs['poses_x'],
                              obs['poses_y'],
                              obs['poses_theta']]).T.double()

        speed, steer, prim = ego_controller.plan(poses[0])

        end = time.time()

        # print(end-start)

        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))

        planners = [(ego_controller.MP.x, ego_controller.MP.y, prim)]
        env.render(mode='human_fast', planner_data=planners)


