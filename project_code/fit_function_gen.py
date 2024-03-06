import numpy as np
import torch

# TODO: Add more functions to test fitting
def gen_spiral_data(t_min, t_max, num_points):
    t = np.linspace(t_min, t_max, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    spiral_data = np.stack((x, y), axis = -1)
    spiral_tensor = torch.from_numpy(spiral_data).float()
    
    return t, spiral_tensor