import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchdiffeq import odeint_adjoint as odeint

# Assume your CSV has columns "Time (min)" and "100 ml Temperature °C" for simplicity
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    t_span = torch.tensor(data['Time (min)'].values, dtype=torch.float)  # Directly use times from CSV
    temperatures = torch.tensor(data['100 ml Temperature °C'].values, dtype=torch.float).unsqueeze(1)
    return t_span, temperatures

class NewtonsCoolingLaw(nn.Module):
    def __init__(self, k_init=-0.1, T_env_init=25.0):
        super(NewtonsCoolingLaw, self).__init__()
        self.k = nn.Parameter(torch.tensor([k_init]))  # Initial guess for the cooling constant
        self.T_env = nn.Parameter(torch.tensor([T_env_init]))  # Initial guess for the ambient temperature

    def forward(self, t, T):
        return self.k * (T - self.T_env)

def plot_results(t, true_y, pred_y, title="Newton's Law of Cooling"):
    plt.figure(figsize=(10, 5))
    plt.scatter(t, true_y.numpy(), label='True')  # Remove detach() since true_y is now from CSV
    plt.plot(t, pred_y.detach().numpy(), 'g--', label='Predicted')  
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    plt.show()

def main(csv_file='path_to_your_csv_file.csv'):
    # Load true data from CSV
    t_span, true_trajectory = load_data(csv_file)
    T_initial = true_trajectory[0]  # Initial temperature from data

    # Trainable model
    model = NewtonsCoolingLaw()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for epoch in range(200):
        optimizer.zero_grad()
        predicted_trajectory = odeint(model, T_initial, t_span)
        loss = loss_func(predicted_trajectory, true_trajectory)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
            final_k = model.k.item()
            print("Estimated cooling constant, k:", final_k)
            final_t = model.T_env.item()
            print("Estimated ambient temp, t:", final_t)

    plot_results(t_span, true_trajectory, predicted_trajectory)

if __name__ == "__main__":
    main('Temperature_Data.csv')  # Example CSV file path

    #Epoch 1980: Loss = 5.306647300720215
    #Estimated cooling constant, k: -0.10220076888799667
