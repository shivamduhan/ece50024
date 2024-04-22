import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt 

from ODE_solvers import *
from fit_function_gen import *
from experiments import *

# Use CPU for now, change to GPU later
dev_used = torch.device('cpu')

# Define the ODE Function as a neural net
class ODEFunc(nn.Module):
    def __init__(self, in_dim, hidden_dim = 64):
        super(ODEFunc, self).__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim, in_dim)

    def forward(self, t, y):
        dy_dt = self.fc1(self.tanh(self.fc0(y ** 3)))
        return dy_dt

# Hyperparameters for the training
in_dim = 2       # 2D data
h = 192          # hidden dimension size
batch_size = 2000 # Batch size
num_epochs = 501 # Total epochs of training
lr = 1e-2        # Learning rate

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
    # Train a NeuralODE using backprop through the ODESovler
    # We use spirals similar to the paper
    # experiment2(dev_used, ODEFunc, in_dim, num_epochs, lr, batch_size, num_points)
    
    # Experiment 3
    # Train using adjoint method
    experiment3(dev_used, ODEFunc, in_dim, num_epochs, lr, batch_size, num_points)
    
    # Adjoint methodcode, TODO: fix
    '''
    # Training loop
    # Create the model, solver, loss function, and optimizer
    ode_func = ODEFunc(in_dim)
    solver = ODESolver(ode_func)
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(ode_func.parameters(), lr = lr)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Get predictions
        z_0 = true_data[0]
        pred_data = solver.forward(z_0, t_eval, dt)

        loss = loss_fn(pred_data, true_data)
        
        # Calculate the initial gradient dLdz using autograd
        z_T = pred_data[-1]
        z_T.requires_grad_(True)
        loss_at_T = loss_fn(z_T, true_data[-1])
        dLdz_T = torch.autograd.grad(loss_at_T, z_T)[0]

        # Iterate over time points in reverse order
        for t in range(len(t_eval) - 2, 0, -1):
            # Calculate the gradients using adjoint method
            dLdz_T, dLdp = adjoint_solve(ode_func, pred_data[t], t_eval[t: t + 2], tuple(ode_func.parameters()), dLdz_T, dt)

            # Accumulate gradients for each model parameter
            for param, grad in zip(ode_func.parameters(), dLdp):
                param.grad = param.grad + grad if param.grad is not None else grad
        # Normalize gradients
        for param in ode_func.parameters():
            if param.grad is not None:
                param.grad /= len(t_eval)
        
        loss.backward()
        optimizer.step()
        
        
        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    '''