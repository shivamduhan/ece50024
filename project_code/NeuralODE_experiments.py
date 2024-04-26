import numpy as np
import matplotlib.pyplot as plt
import csv 

import torch
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
    spiral_matrix = torch.Tensor([[-0.2, 1.], [-1., -0.2]]) # Matrix that describes the generated spiral
    z_init = Variable(torch.Tensor([[-2.0, -2.0]]))         # Initial starting point for the spiral

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
    # Use Adam optimizer
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr = 0.01)
    # Run the training for specified number of epochs
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
    
# Real-world Neural ODE application to Netwon's law of cooling
def experiment3(n_epoch, batch_size, plot_freq, true_data_file, file_name):
    # Get the true data first
    time_data = []
    temp_data_100_ml = []
    temp_data_300_ml = []
    temp_data_800_ml = []
    with open(true_data_file) as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            time_data.append(float(row[0]))
            temp_data_100_ml.append(float(row[1]))
            temp_data_300_ml.append(float(row[2]))
            temp_data_800_ml.append(float(row[3]))

    # Convert lists to PyTorch tensors
    time_tensor = torch.tensor(time_data)
    temp_tensor_100_ml = torch.tensor(temp_data_100_ml)
    temp_tensor_300_ml = torch.tensor(temp_data_300_ml)
    temp_tensor_800_ml = torch.tensor(temp_data_800_ml)
        
    # Set the ode solver to the predetermined ODESolver object
    ode_solver = ODESolver()
    
    # Cooling ODE function to structure (will be used in our neural net) 
    # Trainable parameters: k and ambient temperature
    class CoolingODEF(ODEFunction):
        def __init__(self, k, temp_ambient):
            super(CoolingODEF, self).__init__()
            self.k = nn.Parameter(k)
            self.temp_ambient = nn.Parameter(temp_ambient)

        def forward(self, T, t):
            return self.k * (T - self.temp_ambient)

    # Random initial guess for function (random k value and ambient temperature)
    class TrainCoolingODEF(CoolingODEF):
        def __init__(self):
            super(TrainCoolingODEF, self).__init__(torch.randn(1), torch.randn(1) + 18)

    # Train different models for all volumes
<<<<<<< HEAD
=======
    # volumes = [100, 300, 800]
>>>>>>> 6fc2d1a51f6630bc6c05aae9e316b4fa7627ec9e
    volumes = [300,]
    temp_tensors = [temp_tensor_100_ml, temp_tensor_300_ml, temp_tensor_800_ml]
    
    for vol, temp_tensor in zip(volumes, temp_tensors):
        # Neural ODE for training, use Adam optim
        ode_trained = NeuralODE(TrainCoolingODEF())
        optimizer = torch.optim.Adam(ode_trained.parameters(), lr = 0.05)
        temp_init = Variable(torch.Tensor(temp_tensor[0])).view(-1, 1)
        # For saving the loss
        loss_file = open(f'loss_log_{vol}ml.csv', 'w', newline = '')
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow(['Epoch', 'Loss'])
        
        # Run the training for specified number of epochs
        for i in range(n_epoch):
            # Batch data
            temp_batch, time_batch = create_batch_newtons(temp_tensor, time_data, time_tensor, time_data[-1], np.arange(len(time_data)), batch_size)
            # Predict and update model
            pred_data = ode_trained(temp_init, time_batch, save_all = True, ode_solver = ode_solver)
            loss = F.mse_loss(pred_data, temp_batch.detach())
            optimizer.zero_grad()
            loss.backward(retain_graph = True)  # Uses Adjoint method
            optimizer.step()
            loss_writer.writerow([i, loss.item()])

            # Plot
            if i % plot_freq == 0:
                full_prediction = ode_trained(temp_init, time_tensor, save_all = True, ode_solver = ode_solver)
                plot_newtons_data_results(temp_tensor, time_tensor, full_prediction, vol, i, file_name)
                print(f'Volume: {vol} ml, Epoch: [{i}/{n_epoch}], Loss: {loss.item():.4f}, {[_ for _ in ode_trained.parameters()]}')

        loss_file.close()

        # Wanna see final k and temp
        param_file = open(f'trained_params_{vol}ml.txt', 'w')
        for name, param in ode_trained.named_parameters():
            param_file.write(f'{name}: {param.data}\n')
        param_file.close()