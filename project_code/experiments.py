import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 

from ODE_solvers import *
from fit_function_gen import *

# Solving an ODE for a pre-defined function
def experiment1(t_eval):
    # Function to test (we want to find the derivative of that function)
    func_test = sin_cos_comb(2, 4)
    derivative_vals = func_test(t_eval, torch.tensor([0]))
    y_init = torch.tensor([-2]) # We're solving an IVP, so we know the first initial value
    # Solve the ODE
    solver = ODESolver(func_test)
    solved_ODE = solver(y_init, t = t_eval)
    
    # Plot
    # Create a figure and two subplots
    fig, axs = plt.subplots(2, 1, figsize = (8, 6))

    # Plot solved ODE on the first subplot
    axs[0].plot(t_eval, solved_ODE)
    axs[0].set_title('Solved ODE')

    # Plot derivative values on the second subplot
    axs[1].plot(t_eval, derivative_vals, color = 'r')
    axs[1].set_title('Derivative Values')

    # Add labels and adjust layout
    plt.xlabel('Time')
    axs[0].set_ylabel('Function Value')
    axs[1].set_ylabel('Function Value')
    plt.tight_layout()
    axs[0].grid()
    axs[1].grid()
    
    # Show the plot
    plt.savefig(fname= "./pictures/exp1.png", format = "png")
    plt.close()
    return

# Training a NeuralODE using backprop
def experiment2(dev_used, ODEFunc, in_dim, num_epochs, lr, batch_size, num_points):
    # Generate training data (a spiral)
    true_y0 = torch.tensor([[2.0, 0.0]]).to(dev_used)    # Starting point of the path
    t_new = np.linspace(0., 40., num_points)             # Times at which to evaluate the path
    matrix_spiral = torch.tensor([[-0.1, 5.0], [-5.0, -3.0]]).to(dev_used) # Matrix describing the spiral
    loss_ls = []                                         # Stores losses at each iteration

    # To create the spiral matrix the ODE is described by matmul of y**3 and the matrix describing the spiral
    def true_func(t, y):
        return torch.mm(y ** 3, matrix_spiral)

    # Create the ground truth matrix by solving the ODE for the spiral
    with torch.no_grad():
        solver = ODESolver(true_func)
        true_y = solver(true_y0, t_new)

    # Plot for the original spiral data (ground truth)
    plt.plot(true_y[:, 0].cpu(), true_y[:, 1].cpu())
    plt.title('Original Spiral')
    plt.grid()
    plt.savefig('./pictures/orig_spiral.png')
    plt.close()

    # Training
    # Vars for plotting at specific times
    show_epochs = [1, 10, 50, 500]
    epoch_iter = 0
    # Training objects
    ode_func = ODEFunc(in_dim).to(dev_used)    # Neural net (the f)
    solver = ODESolver(ode_func).to(dev_used)  # ODE solver object
    optimizer = optim.RMSprop(solver.parameters(), lr = lr) # optimizer, use RMS prop
    loss_fn = nn.MSELoss()                                  # loss, use MSE
    
    # Train for a number of epochs
    for epoch in range(num_epochs):
        # Plot at different rounds of training
        if epoch in show_epochs:
            # For the final prediction, predict for longer
            if epoch_iter == 3:
                t_new = np.linspace(0., 60., num_points) 
            pred_y = solver(true_y0, t_new)
            plt.plot(true_y[:, 0].cpu().detach().numpy(), true_y[:, 1].cpu().detach().numpy(), label = 'True Data')
            plt.plot(pred_y[:, 0].cpu().detach().numpy(), pred_y[:, 1].cpu().detach().numpy(), label = 'Predicted Data')
            plt.title(f'Predicted Spiral Data after {show_epochs[epoch_iter]} epochs of training')
            plt.legend()
            plt.grid()
            plt.savefig(f'./pictures/pred_spiral_{show_epochs[epoch_iter]}.png')
            plt.close()
            epoch_iter += 1
            
        # Reset the optimizer gradients to 0 befroe any backprop
        optimizer.zero_grad()
        # Get a batch of times and y values to test for
        # batch_t = list(sorted(t_new[np.random.choice(t_new.shape[0], size = batch_size, replace = False)]))
        batch_t = t_new
        batch_y = true_y
        # Get the predict for these times starting at the ground truth (since IVP, we always have y0)
        pred_y = solver(true_y0, batch_t)
        # Calculate the loss and backprop
        loss = loss_fn(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        # For saving the loss
        loss_ls.append(loss.cpu().item())
        # Print losses once every epoch
        if (epoch) % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}, Training Loss: {loss.item():.4f}")

    # Plot the loss curve
    plt.plot(np.arange(len(loss_ls)), loss_ls, label = 'loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('./pictures/loss_spiral.png')
    plt.close()
    
def experiment3(dev_used, ODEFunc, in_dim, num_epochs, lr, batch_size, num_points):
    # Generate training data (a spiral)
    true_y0 = torch.tensor([[2.0, 0.0]]).to(dev_used)    # Starting point of the path
    t_new = np.linspace(0., 40., num_points)             # Times at which to evaluate the path
    matrix_spiral = torch.tensor([[-0.1, 5.0], [-5.0, -3.0]]).to(dev_used) # Matrix describing the spiral
    loss_ls = []                                         # Stores losses at each iteration

    # To create the spiral matrix the ODE is described by matmul of y**3 and the matrix describing the spiral
    def true_func(t, y):
        return torch.mm(y ** 3, matrix_spiral)

    # Create the ground truth matrix by solving the ODE for the spiral
    with torch.no_grad():
        solver = ODESolver(true_func)
        true_y = solver(true_y0, t_new)

    # Plot for the original spiral data (ground truth)
    plt.plot(true_y[:, 0].cpu(), true_y[:, 1].cpu())
    plt.title('Original Spiral')
    plt.grid()
    plt.savefig('./pictures/orig_spiral.png')
    plt.close()

    # Training
    # Vars for plotting at specific times
    show_epochs = [1, 10, 50, 500]
    epoch_iter = 0
    # Training objects
    ode_func = ODEFunc(in_dim).to(dev_used)    # Neural net (the f)
    solver = ODESolver(ode_func).to(dev_used)  # ODE solver object
    optimizer = optim.RMSprop(solver.parameters(), lr = lr) # optimizer, use RMS prop
    loss_fn = nn.MSELoss()                                  # loss, use MSE
    
    # Train for a number of epochs
    for epoch in range(num_epochs):
        # Plot at different rounds of training
        if epoch in show_epochs:
            # For the final prediction, predict for longer
            if epoch_iter == 3:
                t_new = np.linspace(0., 60., num_points) 
            pred_y = solver(true_y0, t_new)
            plt.plot(true_y[:, 0].cpu().detach().numpy(), true_y[:, 1].cpu().detach().numpy(), label = 'True Data')
            plt.plot(pred_y[:, 0].cpu().detach().numpy(), pred_y[:, 1].cpu().detach().numpy(), label = 'Predicted Data')
            plt.title(f'Predicted Spiral Data after {show_epochs[epoch_iter]} epochs of training')
            plt.legend()
            plt.grid()
            plt.savefig(f'./pictures/pred_spiral_{show_epochs[epoch_iter]}.png')
            plt.close()
            epoch_iter += 1
            
        # Reset the optimizer gradients to 0 befroe any backprop
        optimizer.zero_grad()
        # Get a batch of times and y values to test for
        # batch_t = list(sorted(t_new[np.random.choice(t_new.shape[0], size = batch_size, replace = False)]))
        batch_t = t_new
        batch_y = true_y
        # Get the predict for these times starting at the ground truth (since IVP, we always have y0)
        pred_y = solver(true_y0, batch_t)
        
        # Calculate the grads using adjoint
        loss = loss_fn(pred_y, batch_y)

        # Calculate the initial gradient dLdz using autograd
        z_T = pred_y[-1]
        z_T.requires_grad_(True)
        loss_at_T = loss_fn(z_T, batch_y[-1])
        dLdz_T = torch.autograd.grad(loss_at_T, z_T)[0]
        print("dLdz_T:", dLdz_T)
        # Iterate over time points in reverse order
        for t in range(len(batch_t) - 2, -1, -1):
            # Calculate the gradients using adjoint method
            dt = np.abs(batch_t[t + 1] - batch_t[t])
            dLdz_T, dLdp = adjoint_solve(ode_func, pred_y[t], batch_t[t: t + 2], tuple(ode_func.parameters()), dLdz_T, dt)
            print("dLdz_T:", dLdz_T)
            print("dLdp:", dLdp)
            # Accumulate gradients for each model parameter
            for param, grad in zip(ode_func.parameters(), dLdp):
                param.grad = param.grad + grad if param.grad is not None else grad

        # Normalize gradients
        for param in ode_func.parameters():
            if param.grad is not None:
                param.grad /= len(batch_t)

        optimizer.step()
        # For saving the loss
        loss_ls.append(loss_at_T.cpu().item())
        # Print losses once every epoch
        if (epoch) % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}, Training Loss: {loss.item():.4f}")

    # Plot the loss curve
    plt.plot(np.arange(len(loss_ls)), loss_ls, label = 'loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('./pictures/loss_spiral.png')
    plt.close()
