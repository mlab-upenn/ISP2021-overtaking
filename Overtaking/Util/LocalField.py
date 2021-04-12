
import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_against_data(data, resolution, local_scale,
                            data_scale, ego_pose, data_pose, null_value):
    """
    data: the 2d grid of data that the local field should sample from
    resolution: number of squares in local grid
    local_relative_scale: size of the local grid/size of the data grid
    data_scale: scale of the data in world space
    pose_delta: opp_pose - ego_pose in local space
    """

    relative_scale = local_scale/data_scale

    local_pose = get_local_pose(ego_pose, data_pose)

    data_offset = local_pose[:2] / (local_scale / 2) #+ torch.tensor([1, 0])

    #x,y in local space is y,x in data space
    data_offset[[0,1]] = data_offset[[1,0]]
    aff_g = ((generate_local_affine_grid(local_pose[2], resolution) - data_offset ))* relative_scale

    print(ego_pose)
    if(ego_pose[1] > 5):
        plt.scatter(aff_g[:,:,:,0], aff_g[:,:,:,1])

        plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1], 'r')
        graph_pose = -data_offset*relative_scale
        plt.scatter(graph_pose[0], graph_pose[1])
        plt.show()
    sampled_data = torch.nn.functional.grid_sample(data, aff_g.float())
    off_mask = torch.any(torch.abs(aff_g) > 1, 3).squeeze()
    sampled_data = sampled_data.squeeze()
    sampled_data[off_mask] = null_value
    return sampled_data


def sample_against_data2(data, resolution, local_scale,
                            data_scale, ego_pose, data_pose, null_value):
    """
    data: the 2d grid of data that the local field should sample from
    resolution: number of squares in local grid
    local_relative_scale: size of the local grid/size of the data grid
    data_scale: scale of the data in world space
    pose_delta: opp_pose - ego_pose in local space
    """
    opp_rel_pose_world = get_local_pose(data_pose, ego_pose)
    ego_rel_pose_data = opp_rel_pose_world[:2] / (local_scale/2)

    sample_grid = generate_local_affine_grid(opp_rel_pose_world[2], resolution)

    # sample_grid = sample_grid - torch.tensor([torch.cos(-opp_rel_pose_world[2]), torch.sin(-opp_rel_pose_world[2])])


    sample_grid = (sample_grid + ego_rel_pose_data)*(local_scale/data_scale) - torch.tensor([1,0]).to(sample_grid.device)

    sampled_data = torch.nn.functional.grid_sample(data, sample_grid.float())
    off_mask = torch.any(torch.abs(sample_grid) > 1, 3).squeeze()
    sampled_data = sampled_data.squeeze()
    sampled_data[off_mask] = null_value

    sampled_data = sampled_data.transpose(0,1)
    return sampled_data


def generate_local_affine_grid(theta, resolution):
    """
    theta: world angle for the car
    resolution: number of squares in grid
    """
    T = torch.tensor([[torch.cos(theta), -torch.sin(theta), torch.cos(theta)],
                        [torch.sin(theta), torch.cos(theta), torch.sin(theta)]]).unsqueeze(0)
    return torch.nn.functional.affine_grid(T, torch.Size((1, 1, resolution + 1, resolution + 1)))

def get_local_pose(ego_pose, pose):
    """
    ego_pose: pose of the ego car in x,y,theta
    pose: pose of the point in world space

    return: local x,y of point in ego frame
    """
    ego_pose = ego_pose.double()
    pose = pose.double()
    transform = torch.tensor([[torch.cos(ego_pose[2]), -torch.sin(ego_pose[2]), ego_pose[0]],[ torch.sin(ego_pose[2]), torch.cos(ego_pose[2]), ego_pose[1]], [0,0,1]])
    position = torch.matmul(torch.inverse(transform), torch.tensor([pose[0], pose[1], 1]))[0:2]

    return torch.tensor([position[0], position[1], pose[2] - ego_pose[2]])


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


