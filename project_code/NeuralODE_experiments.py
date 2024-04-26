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

use_cuda = torch.cuda.is_available()

from NeuralODE_code import *
from fit_function_gen import *

# Use CPU for now, change to GPU later
dev_used = torch.device('cuda')

# Solving an ODE for a pre-defined function
def experiment1(t_eval):
    # Function to test (we want to find the derivative of that function)
    func_test = sin_cos_comb(2, 4)
    derivative_vals = func_test(torch.tensor([0]), t_eval)
    y_init = torch.tensor([-2])  # We're solving an IVP, so we know the first initial value
    
    # Solve the ODE
    solver = ODESolver()
    solved_ODE = [y_init]
    y = y_init
    for i in range(len(t_eval) - 1):
        y = solver(y, t_eval[i], t_eval[i + 1], func_test)
        solved_ODE.append(y)
    solved_ODE = torch.cat(solved_ODE)
    # Plot

    # Create a figure and two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot solved ODE on the first subplot
    axs[0].plot(t_eval, solved_ODE)
    axs[0].set_title('Solved ODE')

    # Plot derivative values on the second subplot
    axs[1].plot(t_eval, derivative_vals, color='r')
    axs[1].set_title('Derivative Values')

    # Add labels and adjust layout
    plt.xlabel('Time')
    axs[0].set_ylabel('Function Value')
    axs[1].set_ylabel('Function Value')
    plt.tight_layout()
    axs[0].grid()
    axs[1].grid()
    
    # Show the plot
    plt.savefig(fname="./pictures/RK4_solver_test.png", format="png")
    plt.close()
    return

# Training NeuralODE to fit a spiral 
def experiment2(n_epoch, batch_size, plot_freq, file_name):
    # First set the ode solver to the predetermined ODESolver object
    ode_solver = ODESolver()                                # ODE Solver
    # Spiral parameters
    spiral_matrix = torch.Tensor([[-0.1, 2.], [-2., -0.1]]) # Matrix that describes the generated spiral
    z_init = Variable(torch.Tensor([[-4.0, -2.0]]))         # Initial starting point for the spiral

    # Linear ODE function to structure (will be used in our neural net) 
    class LinearODEF(ODEFunction):
        def __init__(self, W):
            super(LinearODEF, self).__init__()
            self.lin = nn.Linear(2, 2, bias = False)
            self.lin.weight = nn.Parameter(W)

        def forward(self, x, t):
            return self.lin(x)

    # True function for generating the data (spiral)
    class SpiralFunction(LinearODEF):
        def __init__(self):
            super(SpiralFunction, self).__init__(spiral_matrix)
            
    # Random initial guess for function (random weights, correct structure)
    class TrainLinearODEF(LinearODEF):
        def __init__(self):
            super(TrainLinearODEF, self).__init__(torch.randn(2, 2)/2.)

    # Define the Neural ODEs with the predefined function structure
    # One for creating true data
    # One for training to find correct parameters
    ode_true = NeuralODE(SpiralFunction())
    ode_trained = NeuralODE(TrainLinearODEF())
    
    # Create data
    t_max = 8 * np.pi                               # Max time for generating the path
    n_points = 200                                  # Number of points generated in the path

    # Index and times numpy data
    index_np = np.arange(0, n_points, 1, dtype = np.int64)
    index_np = np.hstack([index_np[:, None]])
    times_np = np.linspace(0, t_max, num = n_points)
    times_np = np.hstack([times_np[:, None]])
    
    # Then tensor times data, and generated observation data
    times = torch.from_numpy(times_np[:, :, None]).to(z_init)
    observations = ode_true(z_init, times, save_all = True, ode_solver = ode_solver).detach()
    observations = observations + torch.randn_like(observations) * 0.01 # Add some randomness to the data

    # Train Neural ODE
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr = 0.01)
    for i in range(n_epoch):
        # Get the batched data
        observation_batch, time_batch = create_batch(observations, times_np, times, t_max, index_np, batch_size)
        # Predict and update model
        pred_data = ode_trained(observation_batch[0], time_batch, save_all = True, ode_solver = ode_solver)
        loss = F.mse_loss(pred_data, observation_batch.detach())
        optimizer.zero_grad()
        loss.backward(retain_graph = True) # Uses Adjoint method
        optimizer.step()

        # Plot ever so often
        if i % plot_freq == 0:
            full_prediction = ode_trained(z_init, times, save_all = True, ode_solver = ode_solver)
            plot_ODE_sol(observations = [observations], times = [times], pred_path = [full_prediction], figname = f'./pictures/{file_name}_{i}.png')
            print(f'Epoch: [{i}/{n_epoch}], loss: {loss}')
    
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
    experiment2(n_epoch = N_EPOCH, batch_size = BATCH_SIZE, plot_freq = plot_freq, file_name = "example_spiral/spiral_fit")