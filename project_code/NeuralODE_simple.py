import torch 
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from ODE_solvers import *
from fit_function_gen import *

dev_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the ODE Function as an example MLP
class ODEFunc(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.relu = nn.ReLU()

    def forward(self, t, y):
        y = self.relu(self.fc0(y))
        y = self.relu(self.fc1(y))
        dy_dt = self.fc2(y)
        return dy_dt


# Hyperparameters
in_dim = 2  # MNIST image size (28x28)
h = 128
batch_size = 50
num_epochs = 1000
lr = 3e-4

if __name__ == '__main__':
    # Generate training data
    t_min = 0
    t_max = 6 * np.pi
    num_points = 200

    t_eval, true_data = gen_spiral_data(t_min, t_max, num_points)

    # Training loop
    # Create the model, solver, loss function, and optimizer
    ode_func = ODEFunc(in_dim, h)
    solver = ODESolver(ode_func)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ode_func.parameters(), lr = lr)

    # Training loop
    for epoch in range(num_epochs):
        # Get predictions
        z_0 = true_data[0]
        pred_data = solver.solve(z_0, t_eval)
        loss = loss_fn(pred_data, true_data)
        optimizer.zero_grad()
        # Gradient of loss with respect to last output = d(z_T - z^hat_T)^2 / N = = 2 (z_T - z^hat_T) / B
        dLdz_T = 2 * (pred_data[-1] - true_data[-1]) / num_points
        # Calculate the gradients using adjoint method 
        dLdz_0, dLdp = adjoint_solve(ode_func, pred_data[-1], t_eval, tuple(ode_func.parameters()), dLdz_T)
        # Update the model parameters (set gradients and step)
        for param, grad in zip(ode_func.parameters(), dLdp):
            param.grad = grad
        optimizer.step()

        # Print the loss for every 100 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
