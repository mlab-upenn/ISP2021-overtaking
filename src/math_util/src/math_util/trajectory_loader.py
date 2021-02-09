
import numpy as np
import geometric_types as geo_t

def create_trajectory_from_file(path):
    path = np.genfromtxt(path, delimiter=',')
    trajectory = []

    for i,row in enumerate(path):
        seg = geo_t.line_segment(row, path[(i+1)%path.shape[0]])
        trajectory.append(seg)

    return trajectory