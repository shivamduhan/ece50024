import torch
import torch.nn as nn
import numpy as np

# TODO: implement more solvers
# Runs a single step of RK4 ODE solver to get from y_prev to y_next (y_next can be back in time)
def rk4_step(func, y_prev, t, dt):
    if not isinstance(y_prev, tuple):
        k_1 = func(t, y_prev)
        k_2 = func(t + dt / 2, y_prev + dt * k_1 / 2)
        k_3 = func(t + dt / 2, y_prev + dt * k_2 / 2)
        k_4 = func(t + dt, y_prev + dt * k_3)
        dy = dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        return dy
    else:
        # First element follows normally
        k_1 = func(t, y_prev)
        new_input = tuple([y_prev_elem + dt * k_1_elem / 2 for y_prev_elem, k_1_elem in zip(y_prev, k_1)])
        k_2 = func(t + dt / 2, new_input)
        new_input = tuple([y_prev_elem + dt * k_2_elem / 2 for y_prev_elem, k_2_elem in zip(y_prev, k_2)])
        k_3 = func(t + dt / 2, new_input)
        new_input = tuple([y_prev_elem + dt * k_3_elem for y_prev_elem, k_3_elem in zip(y_prev, k_3)])
        k_4 = func(t + dt, new_input)
        y_next = tuple([y_prev_elem + dt / 6 * (k_1_elem + 2 * k_2_elem + 2 * k_3_elem + k_4_elem) for y_prev_elem, k_1_elem, k_2_elem, k_3_elem, k_4_elem in zip(y_prev, k_1, k_2, k_3, k_4)])

    return y_next

# Solves the desired ODE
class ODESolver(nn.Module):
    def __init__(self, func):
        super(ODESolver, self).__init__()
        self.func = func

    def forward(self, y_init, t):
        # For now use this for backward pass for augmented state
        if isinstance(y_init, tuple):
            # If y_init is a tuple, create a tuple of zero tensors for each element
            y = tuple(torch.zeros((len(t), *element.shape), dtype = element.dtype, device = element.device)
                      for element in y_init)
            for i in range(len(y_init)):
                y[i][0] = y_init[i]

            for i in range(1, len(t)):
                y_prev = tuple(element[i - 1] for element in y)
                y_next = rk4_step(self.func, y_prev, t[i - 1], dt)
                for j in range(len(y)):
                    y[j][i] = y_next[j]
        # For now use this for forward pass for output
        else:
            # Create a tensor of zeros to store the solution
            # y = torch.zeros((len(t), *y_init.shape), dtype = y_init.dtype, device = y_init.device)
            sol = []
            sol.append(y_init)
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                dy = rk4_step(self.func, y_init, t[i - 1], dt)
                y_next = y_init + dy
                sol.append(y_next)
                y_init = y_next
        sol = torch.cat(sol)        
        return sol
        
# Implementation of Algorithm 1: Reverse-mode derivative of an ODE initial value problem from the paper
# Return gradients of loss with respect to parameters for backprop
# func := neural net (function for approximating the behavior)
# z_final := final state of the system at sime final time T
# t := array of time points
# model_prams := theta
# dLdz_T := gradient of the loss with respect to final state
def adjoint_solve(func, z_final, t, model_params, dLdz_T, dt):
    # Initial augmented state s_0
    s_0 = (z_final, dLdz_T) + tuple(torch.zeros(param.shape, dtype = param.dtype, device = param.device) for param in model_params)

    # Define the augmented dynamics as per the algorithm, accept the current time t and the augmented state s
    def aug_dynamics(t, s):
        # Get the current state and the adjoint
        z, a, *_ = s
        # Calculate the vector-Jacobian products
        with torch.enable_grad():
            # First calculate the output of the function given current state and time
            z = z.detach().requires_grad_(True)
            f = func(t, z)

            # Then get the vector-Jacobian products for each part of the augmented state
            # Calculate gradients of f with respect to z, and model parameters
            vjp_z, *vjp_params = torch.autograd.grad(
                f, (z,) + model_params, -a,
                allow_unused = True, retain_graph = True
            )

        # Return the augmented state
        return (f, vjp_z, *vjp_params)

    # Create an instance of the ODESolver with the augmented dynamics
    solver = ODESolver(aug_dynamics)

    # Reverse the time points for solving the augmented ODE backward in time
    t_reversed = torch.flip(torch.tensor(t), [0])
    # Solve the augmented ODE backward in time using the ODESolver
    print(s_0)
    print("\n")
    print(t_reversed)
    print("\n")
    print(dt)
    s_T = solver.forward(s_0, t_reversed, dt)
    # Extract the final augmented state
    _, adj_z_T, *adj_params_T = s_T
    adj_z_T = adj_z_T[-1]
    for i, param in enumerate(adj_params_T):
        adj_params_T[i] = param[-1]

    # Extract the gradients from the final augmented state
    dLdz_0 = adj_z_T
    dLdp = [param.view_as(model_param) for param, model_param in zip(adj_params_T, model_params)]
    return dLdz_0, dLdp
