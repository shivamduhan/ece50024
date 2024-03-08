import numpy as np
import torch
import torch.nn as nn

# Sinuosoidal function for experiment 1
class sin_cos_comb(nn.Module):
    def __init__(self, a, b):
        super(sin_cos_comb, self).__init__()
        self.a = a
        self.b = b
    def forward(self, t, y):
        return self.a * torch.sin(t) + self.b * torch.cos(t)

# TODO: Add more functions to test fitting
def gen_spiral_data(t_min, t_max, num_points):
    t = np.linspace(t_min, t_max, num_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    spiral_data = np.stack((x, y), axis = -1)
    spiral_tensor = torch.from_numpy(spiral_data).float()
    
    return t, spiral_tensor



