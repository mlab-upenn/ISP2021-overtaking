import numpy as np

def pareto_front(costs):
    """
    Find the pareto-efficient points from a set of cost data
    
    params:
        costs: An (n_points, n_costs) array
    
    return: An (n_efficient_points, ) integer array of indices of 
        pareto-efficient points.
    """
    n_points = costs.shape[0]
    is_efficient = np.arange(n_points)
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < n_points:
        nondominated_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_mask] # Remove dominated points
        costs = costs[nondominated_mask]
        next_point_index = np.sum(nondominated_mask[:next_point_index]) + 1
    return is_efficient
