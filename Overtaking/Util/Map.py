import torch
import yaml
import numpy as np
from PIL import Image
from argparse import Namespace

class Map():
    def __init__(self, config_yaml_path):
        # open map config file
        with open(config_yaml_path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)
        self.path = conf.map_path
        self.ext = conf.map_ext
        self.start_pose = np.array([conf.sx, conf.sy, conf.stheta])

        # open map
        with open(self.path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as error:
                print(error)

        # load map image
        self.image = torch.tensor(np.array(Image.open(self.path + self.ext)
                        .transpose(Image.FLIP_TOP_BOTTOM))
                        .astype(np.float64)).unsqueeze(0).unsqueeze(0)
        self.height = self.image.shape[2]
        self.width = self.image.shape[3]

    def sample_obstacles(self, pose, local_grid_size, resolution):
        sample_grid = Map._generate_local_affine_grid(pose[2], resolution)
        map_size = self.resolution * torch.tensor([self.width, self.height])
        scaled_position = (torch.tensor([pose[0], pose[1]]) - \
                           torch.tensor([self.origin[0], self.origin[1]])) \
                          / (map_size / 2)
        sample_grid = sample_grid * (local_grid_size / map_size) \
                      - torch.tensor([1, 1]) + scaled_position
        local_grid = torch.nn.functional.grid_sample(self.image, 
                                                     sample_grid, 
                                                     mode='nearest')
        local_grid = torch.round(local_grid / 255)
        local_grid = -(local_grid - 1).squeeze()
        return local_grid

    def sample_against_data(self, data, resolution, local_relative_scale, data_scale, pose_delta, null_value):
        #data: the 2d grid of data that the local field should sample from
        #resolution: number of squares in local grid
        #local_relative_scale: size of the local grid/size of the data grid
        #data_scale: scale of the data in world space
        #pose_delta: difference between poses (x, y, theta)
        data_offset = pose_delta[:2] / (data_scale / 2) - torch.tensor([1, 0])
        aff_g = Map._generate_local_affine_grid(pose_delta[2], resolution) \
                * local_relative_scale + data_offset
        sampled_data = torch.nn.functional.grid_sample(data, aff_g.float())
        off_mask = torch.any(torch.abs(aff_g) > 1, 3).squeeze()
        sampled_data = sampled_data.squeeze()
        sampled_data[off_mask] = null_value
        return sampled_data

    @staticmethod
    def _generate_local_affine_grid(theta, resolution):
        #theta: world angle for the car
        #resolution: number of squares in grid
        theta = torch.tensor(theta).double()
        T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)],
                        [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
        sample_grid = torch.nn.functional.affine_grid(T, torch.Size((1, 1, resolution + 1, resolution + 1)))

        return sample_grid


if __name__ == '__main__':
    Map('config_example_map_filled.yaml')