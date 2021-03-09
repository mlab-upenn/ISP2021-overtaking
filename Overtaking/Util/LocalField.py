
import torch
import numpy as np


def sample_map_obstacles(x, y, theta, map_image, local_grid_world_size, resolution, map_resolution, map_width, map_height, map_origin ):
    x = torch.tensor(x).double()
    y = torch.tensor(y).double()
    theta = torch.tensor(theta).double()
    T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)],
                      [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
    sample_grid = torch.nn.functional.affine_grid(T, torch.Size((1, 1, resolution + 1, resolution + 1)))

    map_size = torch.tensor([map_resolution * map_width, map_resolution * map_height])

    scaled_position = (torch.tensor([x, y]) - torch.tensor([map_origin[0], map_origin[1]])) / (map_size / 2)

    # print(scaled_position)
    sample_grid = sample_grid * (local_grid_world_size / (map_size)) - torch.tensor([1, 1]) + scaled_position
    # print(sample_grid)

    local_grid = torch.nn.functional.grid_sample(map_image, sample_grid, mode='nearest')
    local_grid = torch.round(local_grid / 255)

    local_grid = -(local_grid - 1).squeeze()

    return local_grid