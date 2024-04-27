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
    # experiment2(n_epoch = N_EPOCH, batch_size = BATCH_SIZE, plot_freq = plot_freq, file_name = "spiral_fit") # Uncomment to run fitting to a spiral
    
    # Experient 3
    N_EPOCH = 500
    BATCH_SIZE = 32
    plot_freq = 10
    experiment3(n_epoch = N_EPOCH, batch_size = BATCH_SIZE, plot_freq = plot_freq, true_data_file = "./Temperature_Data.csv", file_name = "real_world_example/newtons_law_of_cooling") # Uncomment to run fitting to a spiral
    
    
    # Experient 4
    N_EPOCH = 2000
    BATCH_SIZE = 32
    plot_freq = 10
    # experiment4(n_epoch = N_EPOCH, batch_size = BATCH_SIZE, plot_freq = plot_freq, file_name = "spiral_data_with_DNN") # Uncomment to run fitting to a spiral

    pass