import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchdiffeq import odeint_adjoint as odeint

class NewtonsCoolingLaw(nn.Module):
    def __init__(self, k_init=-0.1, T_env_init=25.0):
        super(NewtonsCoolingLaw, self).__init__()
        self.k = nn.Parameter(torch.tensor([k_init]))  # Initial guess for the cooling constant
        self.T_env = nn.Parameter(torch.tensor([T_env_init]))  # Initial guess for the ambient temperature

    def forward(self, t, T):
        return self.k * (T - self.T_env)

def plot_results(t, true_y, pred_y, title="Newton's Law of Cooling"):
    plt.figure(figsize=(10, 5))
    plt.plot(t, true_y.detach().numpy(), label='True')  
    plt.plot(t, pred_y.detach().numpy(), 'g--', label='Predicted')  
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # Simulation settings
    T_initial = torch.tensor([100.0])  # Initial temperature
    t_span = torch.linspace(0, 10, steps=100)  # Time span for the simulation

    # True model with known parameters
    true_model = NewtonsCoolingLaw(k_init=-0.05, T_env_init=25.0)
    
    # Trainable model
    model = NewtonsCoolingLaw()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for epoch in range(200):
        optimizer.zero_grad()
        predicted_trajectory = odeint(model, T_initial, t_span)
        true_trajectory = odeint(true_model, T_initial, t_span)
        loss = loss_func(predicted_trajectory, true_trajectory)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
            final_k = model.k.item()
            print("Estimated cooling constant, k:", final_k)

    plot_results(t_span, true_trajectory, predicted_trajectory)

if __name__ == "__main__":
    main()
