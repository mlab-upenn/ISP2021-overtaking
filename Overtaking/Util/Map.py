import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import Namespace
from Overtaking.Util.LocalField import generate_local_affine_grid
from numba import njit\


class Map():
    def __init__(self, config_yaml_path):
        # open map config file
        with open(config_yaml_path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        self.path = conf.map_path
        self.ext = conf.map_ext
        self.start_pose = np.array([conf.sx, conf.sy, conf.stheta])
        self.waypoints = np.loadtxt(conf.wpt_path, 
                                    delimiter=conf.wpt_delim, 
                                    skiprows=conf.wpt_rowskip)
        self.waypoints = np.vstack((self.waypoints[:, conf.wpt_xind], 
                                    self.waypoints[:, conf.wpt_yind])).T

        # open map
        with open(self.path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.resolution = map_metadata['resolution']
                self.origin = np.array(map_metadata['origin'])[:2]
            except yaml.YAMLError as error:
                print(error)

        # load map image
        self.image = torch.tensor(np.array(Image.open(self.path + self.ext)
                        .transpose(Image.FLIP_TOP_BOTTOM))
                        .astype(np.float64)).unsqueeze(0).unsqueeze(0)
        self.height = self.image.shape[2]
        self.width = self.image.shape[3]

        # calculate reward
        self.reward = self.image.clone()

        segment_lengths = np.linalg.norm(self.waypoints[1:, :] - self.waypoints[:-1, :], axis=1)
        for i in range(self.width):
            for j in range(self.height):
                if self.reward[0, 0, j, i] > 0:
                    position = self.resolution * np.array([i, j]) + self.origin
                    nearest_point, _, _, segment_idx = Map._nearest_point_on_trajectory(position, self.waypoints)
                    self.reward[0, 0, j, i] = np.sum(segment_lengths[:segment_idx]) + np.linalg.norm(self.waypoints[segment_idx, :] - position)

    def display(self):
        fig, axes = plt.subplots(nrows=2, figsize=(4, 6))
        axes.flat[0].set_title("Occupancy Map")
        im = axes.flat[0].imshow(self.image.squeeze())
        axes.flat[1].set_title("Reward Map")
        # im = axes.flat[1].imshow(255 * self.reward.squeeze() / self.reward.max())
        im = axes.flat[1].imshow(self.reward.squeeze())
        # waypoints = (self.waypoints - self.origin) / self.resolution
        # plt.scatter(waypoints[:, 0], waypoints[:, 1])
        plt.colorbar(im, ax=axes.ravel().tolist())  
        [ax.set_axis_off() for ax in axes.ravel()]
        plt.show(block=True)

    def sample_obstacles(self, pose, local_grid_size, resolution):
        local_obstacles = self.sample_from_map(self.image, pose, local_grid_size, resolution)
        local_obstacles = torch.round(local_obstacles/255)
        local_obstacles = -(local_obstacles - 1).squeeze()
        return local_obstacles

    def sample_reward(self, pose, local_grid_size, resolution):
        local_reward = self.sample_from_map(self.reward, pose, local_grid_size, resolution)
        return local_reward

    def sample_from_map(self, map_image, pose, local_grid_size, resolution):
        sample_grid = generate_local_affine_grid(pose[2], resolution)
        map_size = self.resolution * torch.tensor([self.width, self.height])
        scaled_position = (pose[:2] - self.origin) / (map_size / 2)
        sample_grid = sample_grid * (local_grid_size / map_size) \
                      - torch.tensor([1, 1]) + scaled_position
        local_grid = torch.nn.functional.grid_sample(map_image,
                                                     sample_grid,
                                                     mode='nearest')
        return local_grid



    @staticmethod
    @njit(fastmath=False, cache=True)
    def _nearest_point_on_trajectory(point, trajectory):
        '''
        Return the nearest point along the given piecewise linear trajectory.

        Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
        not be an issue so long as trajectories are not insanely long.

            Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

        point: size 2 numpy array
        trajectory: Nx2 matrix of (x,y) trajectory waypoints
            - these must be unique. If they are not unique, a divide by 0 error will destroy the world
        '''
        diffs = trajectory[1:, :] - trajectory[:-1, :]
        l2s   = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
        # this is equivalent to the elementwise dot product
        # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
        dots = np.empty((trajectory.shape[0] - 1,))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
        t = dots / l2s
        t[t < 0.0] = 0.0
        t[t > 1.0] = 1.0
        # t = np.clip(dots / l2s, 0.0, 1.0)
        projections = trajectory[:-1,:] + (t*diffs.T).T
        # dists = np.linalg.norm(point - projections, axis=1)
        dists = np.empty((projections.shape[0],))
        for i in range(dists.shape[0]):
            temp = point - projections[i]
            dists[i] = np.sqrt(np.sum(temp*temp))
        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


if __name__ == '__main__':
    Map('config_example_map_filled.yaml').display()
    