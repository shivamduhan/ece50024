import torch 
import torch.nn as nn
import torch.optim as optim

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
h = 64
num_epochs = 1000
lr = 3e-4

if __name__ == '__main__':
    # Generate training data
    t_min = 0
    t_max = 6 * np.pi
    num_points = 200
    t_eval, true_data = gen_spiral_data(t_min, t_max, num_points)
    dt = np.abs(t_eval[0] - t_eval[1])

    # Training loop
    # Create the model, solver, loss function, and optimizer
    ode_func = ODEFunc(in_dim, h)
    solver = ODESolver(ode_func)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ode_func.parameters(), lr = lr)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Get predictions
        z_0 = true_data[0]
        pred_data = solver.solve(z_0, t_eval, dt)
        loss = loss_fn(pred_data, true_data)

        # Iterate over time points in reverse order
        for t in range(len(t_eval) - 2, 0, -1):
            dLdz = 2 * (pred_data[t] - true_data[t]) / num_points

            # Calculate the gradients using adjoint method
            _, dLdp = adjoint_solve(ode_func, pred_data[t], t_eval[t:t+2], tuple(ode_func.parameters()), dLdz, dt)

            # Apply gradients to the model parameters
            for param, grad in zip(ode_func.parameters(), dLdp):
                param.grad = grad

            optimizer.step()

        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
