import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

from ODE_solvers import *
from fit_function_gen import *
from experiments import *

torch.manual_seed(0)
np.random.seed(0)

dev_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the ODE Function as an example MLP

class ODEFunc(nn.Module):
    def __init__(self, in_dim, hidden_dim = 64):
        super(ODEFunc, self).__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim, in_dim)

    def forward(self, t, y):
        dy_dt = self.fc1(self.tanh(self.fc0(y)))
        return dy_dt


'''
class ODEFunc(nn.Module):
    def __init__(self, in_dim = 2):
        super(ODEFunc, self).__init__()
        self.fc0 = nn.Linear(in_dim, in_dim, bias = True)
    
    def forward(self, t, x):
        y = self.fc0(x)
        return y
'''
# Hyperparameters
in_dim = 2  # MNIST image size (28x28)
h = 128
num_epochs = 1000 # 1000
lr = 1e-2




if __name__ == '__main__':
    # Generate timing data
    t_min = 0.0
    t_max = 6 * np.pi
    num_points = 2000
    t_eval = torch.linspace(t_min, t_max, num_points)
    dt = np.abs(t_eval[0] - t_eval[1])
    # Experiment 1
    experiment1(t_eval, dt)
    
    # Experiment 2
    # Generate data
    true_y0 = torch.tensor([[2.0, 0.0]])
    t_new = torch.linspace(0., 50., num_points)
    true_A = torch.tensor([[-0.1, 5.0], [-5.0, -2.0]])
    
    class Lambda(nn.Module):
        def forward(self, t, y):
            return torch.mm(y ** 3, true_A)
    
    with torch.no_grad():
        node = ODESolver(Lambda())
        true_y = node(true_y0, t_new, dt)

    plt.plot(true_y[:, 0], true_y[:, 1])
    plt.show()
    
    # Training
    ode_func = ODEFunc(in_dim)
    solver = ODESolver(ode_func)
    
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