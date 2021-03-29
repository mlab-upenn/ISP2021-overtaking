
import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_map_obstacles(x, y, theta, map_image, local_grid_world_size, resolution, map_resolution, map_width, map_height, map_origin ):
    x = torch.tensor(x).double()
    y = torch.tensor(y).double()
    theta = torch.tensor(theta).double()

    sample_grid = generate_local_affine_grid(theta, resolution)

    map_size = torch.tensor([map_resolution * map_width, map_resolution * map_height])

    scaled_position = (torch.tensor([x, y]) - torch.tensor([map_origin[0], map_origin[1]])) / (map_size / 2)

    # print(scaled_position)
    sample_grid = sample_grid * (local_grid_world_size / (map_size)) - torch.tensor([1, 1]) + scaled_position
    # print(sample_grid)

    print(map_image)
    plt.imshow(map_image.squeeze())
    plt.show()

    local_grid = torch.nn.functional.grid_sample(map_image, sample_grid, mode='nearest')
    local_grid = torch.round(local_grid / 255)

    local_grid = -(local_grid - 1).squeeze()

    return local_grid


def sample_against_data(data, resolution, local_relative_scale, data_scale, relative_offset, theta, null_value):
    #data: the 2d grid of data that the local field should sample from
    #resolution: number of squares in local grid
    #local_relative_scale: size of the local grid/size of the data grid
    #data_scale: scale of the data in world space
    #relative_offset: world space offset
    #theta

    data_offset = relative_offset/(data_scale/2) - torch.tensor([1,0])


    aff_g = generate_local_affine_grid(theta, resolution)*local_relative_scale + data_offset

    # print(aff_g)

    sampled_data = torch.nn.functional.grid_sample(data, aff_g.float())

    off_mask = torch.any(torch.abs(aff_g) > 1, 3).squeeze()
    sampled_data = sampled_data.squeeze()
    sampled_data[off_mask] = null_value



    return sampled_data




def generate_local_affine_grid(theta, resolution):
    #theta: world angle for the car
    #resolution: number of squares in grid
    theta = torch.tensor(theta).double()
    T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)],
                      [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
    sample_grid = torch.nn.functional.affine_grid(T, torch.Size((1, 1, resolution + 1, resolution + 1)))

    return sample_grid

if __name__ == '__main__':

    x = 2
    y = 2
    theta = .2

    resolution = 100
    data = torch.ones((resolution+1,resolution+1)).unsqueeze(0).unsqueeze(0)

    local_relative_scale = 1
    data_scale = 5

    relative_offset = torch.tensor([x,y])

    null_value = 0

    sampled_data = sample_against_data(data, resolution, local_relative_scale, data_scale, relative_offset, theta, null_value)

    plt.imshow(sampled_data.squeeze())
    plt.colorbar()
    plt.show()


