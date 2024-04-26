import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F 
from torch.autograd import Variable

from NeuralODE_code import *
from NeuralODE_experiments import * 

# Use CPU for now, change to GPU later
use_cuda = torch.cuda.is_available()
dev_used = torch.device('cuda')    

if __name__ == '__main__':
    # Generate timing data (for our task we generate trajectories over time)
    t_min = 0.0
    t_max = 6 * np.pi
    num_points = 2500
    t_eval = torch.linspace(t_min, t_max, num_points)

    # Experiment 1
    # Test RK4 and ODE solver implementations
    experiment1(t_eval)
    
    # Experiment 2 
    N_EPOCH = 1000
    BATCH_SIZE = 32
    plot_freq = 10
    # experiment2(n_epoch = N_EPOCH, batch_size = BATCH_SIZE, plot_freq = plot_freq, file_name = "example_spiral/spiral_fit") # Uncomment to run fitting to a spiral
    
    # Experient 3
    