
import yaml
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from ..MotionPrimitives.MotionPrimitives import MotionPrimitive


class PureRiskController():

    def __init__(self, map_path, map_ext, local_grid_width = 3, local_grid_height = 3, resolution=50):
        ## from renderer.py in f1tenth gym for loading map image
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # load map image
        self.map_img = np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        ##

        self.resolution = resolution
        self.local_grid_width = local_grid_width
        self.local_grid_height = local_grid_height


    def plan(self, x, y, theta):
        x = torch.tensor(x).double()
        y = torch.tensor(y).double()
        theta = torch.tensor(theta).double()
        T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)], [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
        sample_grid = torch.nn.functional.affine_grid(T, torch.Size((1, 1, self.resolution, self.resolution)))

        # print(sample_grid)

        sg = sample_grid[0].numpy()
        print(sg[:,:,0], sg[:,:,1])
        plt.plot(sg[:,:, 0].transpose(), sg[:, :, 1].transpose(), 'o')
        plt.show()

        # sample_grid = self.grid


