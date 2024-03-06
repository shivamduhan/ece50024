import torch
# TODO: implement more solvers
# Runs a single step of RK4 ODE solver to get from y_prev to y_next (y_next can be back in time)
def rk4_step(func, y_prev, t, dt):
    k1 = func(t, y_prev)
    k2 = func(t + dt / 2, y_prev + dt * k1 / 2)
    k3 = func(t + dt / 2, y_prev + dt * k2 / 2)
    k4 = func(t + dt, y_prev + dt * k3)
    y_next = y_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next

# Solves the desired ODE
class ODESolver:
    def __init__(self, func):
        self.func = func

    def solve(self, y_init, t):
        if isinstance(y_init, tuple):
            # If y_init is a tuple, create a tuple of zeros for each element
            y = tuple(torch.zeros((len(t), *element.shape), dtype=element.dtype, device=element.device)
                      for element in y_init)
            for i in range(len(y_init)):
                y[i][0] = y_init[i]
        else:
            # If y_init is a tensor, create a tensor of zeros
            y = torch.zeros((len(t), *y_init.shape), dtype=y_init.dtype, device=y_init.device)
            y[0] = y_init

        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            if isinstance(y, tuple):
                y_prev = tuple(element[i - 1] for element in y)
                y_next = self.func(t[i - 1], y_prev)
                for j in range(len(y)):
                    y[j][i] = y_next[j]
            else:
                y[i] = rk4_step(self.func, y[i - 1], t[i - 1], dt)

        return y
    
# Implementation of Algorithm 1: Reverse-mode derivative of an ODE initial value problem from the paper
# Return gradients of loss with respect to parameters for backprop
# func := neural net (function for approximating the behavior)
# z_final := final state of the system at sime final time T
# t := array of time points
# model_prams := theta
# dLdz_T := gradient of the loss with respect to final state
def adjoint_solve(func, z_final, t, model_params, dLdz_T):
    # Initial augmented state s_0
    s_0 = (z_final, dLdz_T) + tuple(torch.zeros(param.numel(), dtype = param.dtype, device = param.device) for param in model_params)

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
            _, vjp_z, *vjp_params = torch.autograd.grad(
                f, (z,) + model_params, -a,
                allow_unused=True, retain_graph=True
            )

        # Return the augmented state
        return (f, vjp_z, *vjp_params)

    # Create an instance of the ODESolver with the augmented dynamics
    solver = ODESolver(aug_dynamics)

    # Reverse the time points for solving the augmented ODE backward in time
    t_reversed = torch.flip(torch.tensor(t), [0])

    # Solve the augmented ODE backward in time using the ODESolver
    s_T = solver.solve(s_0, t_reversed)

    # Extract the final augmented state
    _, adj_z_T, *adj_params_T = s_T[-1]

    # Extract the gradients from the final augmented state
    dLdz_0 = adj_z_T
    dLdp = [param.view_as(model_param) for param, model_param in zip(adj_params_T, model_params)]

    return dLdz_0, dLdp