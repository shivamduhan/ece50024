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
def experiment2():
    # First set the ode solver to the predetermined ODESolver object
    ode_solver = ODESolver()
    spiral_matrix = torch.Tensor([[-0.1, 2.], [-2., -0.1]])
    
    def conduct_experiment(ode_true, ode_trained, n_steps, name, plot_freq=10, ode_solver = ode_solver):
        # Create data
        z0 = Variable(torch.Tensor([[-4.0, -2.0]]))

        t_max = 8 * np.pi
        n_points = 200

        index_np = np.arange(0, n_points, 1, dtype=np.int64)
        index_np = np.hstack([index_np[:, None]])
        times_np = np.linspace(0, t_max, num=n_points)
        times_np = np.hstack([times_np[:, None]])

        times = torch.from_numpy(times_np[:, :, None]).to(z0)
        obs = ode_true(z0, times, save_all=True, ode_solver = ode_solver).detach()
        obs = obs + torch.randn_like(obs) * 0.01

        # Get trajectory of random timespan 
        min_delta_time = 1.0
        max_delta_time = 5.0
        max_points_num = 32
        def create_batch():
            t0 = np.random.uniform(0, t_max - max_delta_time)
            t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

            idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

            obs_ = obs[idx]
            ts_ = times[idx]
            return obs_, ts_

        # Train Neural ODE
        optimizer = torch.optim.Adam(ode_trained.parameters(), lr = 0.01)
        for i in range(n_steps):
            obs_, ts_ = create_batch()

            z_ = ode_trained(obs_[0], ts_, save_all=True, ode_solver = ode_solver)
            loss = F.mse_loss(z_, obs_.detach())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if i % plot_freq == 0:
                z_p = ode_trained(z0, times, save_all = True, ode_solver = ode_solver)

                plot_ODE_sol(observations = [obs], times = [times], pred_path = [z_p], figname = f'./pictures/{name}_{i}.png')
                clear_output(wait=True)
                print(f'Epoch: {i}, loss: {loss}')
    
    
    # Restrict ODE to a linear function
    class LinearODEF(ODEFunction):
        def __init__(self, W):
            super(LinearODEF, self).__init__()
            self.lin = nn.Linear(2, 2, bias=False)
            self.lin.weight = nn.Parameter(W)

        def forward(self, x, t):
            return self.lin(x)

    # True function
    class SpiralFunctionExample(LinearODEF):
        def __init__(self):
            super(SpiralFunctionExample, self).__init__(spiral_matrix)
            
    # Random initial guess for function
    class RandomLinearODEF(LinearODEF):
        def __init__(self):
            super(RandomLinearODEF, self).__init__(torch.randn(2, 2)/2.)

    ode_true = NeuralODE(SpiralFunctionExample())
    ode_trained = NeuralODE(RandomLinearODEF())

    conduct_experiment(ode_true, ode_trained, 1000, "spiral_fit_linear")
    
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
    # experiment2()