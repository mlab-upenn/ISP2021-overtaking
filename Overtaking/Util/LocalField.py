
import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_against_data(data, resolution, local_relative_scale, 
                            data_scale, pose_delta, null_value):
    """
    data: the 2d grid of data that the local field should sample from
    resolution: number of squares in local grid
    local_relative_scale: size of the local grid/size of the data grid
    data_scale: scale of the data in world space
    pose_delta: difference between poses (x, y, theta)
    """
    data_offset = pose_delta[:2] / (data_scale / 2) - torch.tensor([1, 0])
    aff_g = generate_local_affine_grid(pose_delta[2], resolution) \
            * local_relative_scale + data_offset
    sampled_data = torch.nn.functional.grid_sample(data, aff_g.float())
    off_mask = torch.any(torch.abs(aff_g) > 1, 3).squeeze()
    sampled_data = sampled_data.squeeze()
    sampled_data[off_mask] = null_value
    return sampled_data

def generate_local_affine_grid(theta, resolution):
    """
    theta: world angle for the car
    resolution: number of squares in grid
    """
    T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)],
                        [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
    return torch.nn.functional.affine_grid(T, torch.Size((1, 1, resolution + 1, resolution + 1)))

if __name__ == '__main__':

    pose_delta = torch.Tensor([2, 2, .2])

    resolution = 100
    data = torch.ones((resolution+1, resolution+1)).unsqueeze(0).unsqueeze(0)

    local_relative_scale = 1
    data_scale = 5

    null_value = 0

    sampled_data = sample_against_data(data, resolution, local_relative_scale, data_scale, pose_delta, null_value)

    plt.imshow(sampled_data.squeeze())
    plt.colorbar()
    plt.show()


