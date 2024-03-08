import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

from ODE_solvers import *
from fit_function_gen import *


def experiment1(t_eval, dt):
    
    # Experiment 1: Check the forward pass first
    # Function to test (we want to find the derivative of that function)
    func_test = sin_cos_comb(2, 4)
    derivative_vals = func_test(t_eval, torch.tensor([0]))
    y_init = torch.tensor([-2]) # We're solving an IVP, so we know the first initial value
    # Solve the ODE
    solver = ODESolver(func_test)
    solved_ODE = solver(y_init, t = t_eval, dt = dt)
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
    axs[0].set_ylabel('Solved ODE')
    axs[1].set_ylabel('Derivative Values')
    plt.tight_layout()

    # Show the plot
    plt.savefig(fname="exp1.png", format="png")
    plt.close()
    expected = -2 * torch.cos(t_eval) + 4 * torch.sin(t_eval)

    print(torch.mean(torch.abs(solved_ODE - expected)))

if __name__ == '__main__':
    t_min = 0.0
    t_max = 6 * np.pi
    num_points = 2000
    t_eval = torch.linspace(t_min, t_max, num_points)
    dt = np.abs(t_eval[0] - t_eval[1])
    experiment1(t_eval, dt)
