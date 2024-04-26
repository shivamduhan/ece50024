import math
import numpy as np

import torch
from torch import nn
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

# Class for solving ODEs using RK-4
class ODESolver(nn.Module):
    def __init__(self):
        super(ODESolver, self).__init__()

    def forward(self, z_start, t_start, t_end, func):
        # Get correct dt for the time bracket
        dt_max = 0.01
        num_steps = math.ceil((abs(t_end - t_start) / dt_max).max().item())
        dt = (t_end - t_start) / num_steps

        # Iterative solver code
        z = z_start
        t = t_start
        
        # Use rk-4 to find z iteratively
        for _ in range(num_steps):
            z = z + self.rk4_step(func, z, t, dt)
            t = t + dt

        # Return final state
        return z

    # Single rk-4 step
    def rk4_step(self, func, y, t, dt):
        k1 = func(y, t)
        k2 = func(y + dt * k1 / 2, t + dt / 2)
        k3 = func(y + dt * k2 / 2, t + dt / 2)
        k4 = func(y + dt * k3, t + dt)
        dy = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return dy
    
# Calculate ODE function output, and gradients wrp to the z (state), t, and model.parameters()
# z = current state of the system (as per paper)
# t = time
# grad_out = dL/dz_prev
class ODEFunction(nn.Module):
    # Calculate all desired values
    def forward_grad(self, z, t, grad_out):
        # Calculate output of the ODE function
        batch_size = z.shape[0]
        out = self.forward(z, t)

        # Calculate the vector-jacobian products (vjp) using adjoint method
        # Use torch.autograd.grad to find grads of output (out) wrp to z, t, and model params
        # Scale by adjoint state (a param from paper)
        adjoint_state = grad_out
        vjp_f_z, vjp_f_t, *vjp_f_params = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs = (adjoint_state),
            allow_unused = True, retain_graph = True
        )
        
        # Adjust the gradients wrp to parameters to match batch size (torch.autograd.grad sums over batches)
        if vjp_f_params:
            # Put into single tensor and expand to match the batch size
            # To match, expand for first dimension and divide by batch size (average grads)
            vjp_f_params = torch.cat([p_grad.flatten() for p_grad in vjp_f_params]).unsqueeze(0)
            vjp_f_params = vjp_f_params.expand(batch_size, -1) / batch_size
        
        # Same for time grads, but no need to flatten since scalar
        if vjp_f_t is not None:
            vjp_f_t = vjp_f_t.expand(batch_size, 1) / batch_size
            
        return out, vjp_f_z, vjp_f_t, vjp_f_params
    
    # Flatten all params into a single tensor
    def flatten_parameters(self):
        flat_parameters = []
        for param in self.parameters():
            flat_parameters.append(param.flatten())
        return torch.cat(flat_parameters)

# Implements the adjoint method for computing the gradients
class ODEAdjoint(torch.autograd.Function):
    # Forward calculation of state using ODE solver
    def forward(backward_context, z_init, t, flat_params, func, ode_solver):
        # Get hyperparams for forward pass
        batch_size, *state_shape = z_init.size()
        num_timesteps = t.size(0)

        # Solve ODE forward in time
        with torch.no_grad():
            # Init the solution tensor to all zeros, set initial to initial state
            z = torch.zeros(num_timesteps, batch_size, *state_shape).to(z_init.device)
            z[0] = z_init
            z_prev = z_init
            # Calculate states using ODE solver (RK-4 in our case)
            for i in range(num_timesteps - 1):
                z_prev = ode_solver(z_prev, t[i], t[i + 1], func)
                z[i + 1] = z_prev

        # Save the necessary variables for backward pass
        backward_context.func = func
        backward_context.save_for_backward(t, z.clone(), flat_params)
        backward_context.solver = ode_solver
        return z

    # Backward calculation of gradients using adjoint method
    # Grad out = dLdz
    def backward(backward_context, grad_out):
        # Get the saved variables
        func = backward_context.func
        t, z, flat_params = backward_context.saved_tensors
        ode_solver = backward_context.solver
        
        # Get hyperparams for adjoint method
        num_timesteps, batch_size, *state_shape = z.size()
        state_dim = np.prod(state_shape)
        num_params = flat_params.size(0)

        # Define the augmented dynamics as per the algorithm, the augmented state aug_z = s and accept the current time t
        def aug_dynamics(aug_z, t_i):
            # Get the current state and the adjoint
            # Only need z(t) (state) and a(t), ignore rest
            z_i, a_i = aug_z[:, : state_dim], aug_z[:, state_dim: 2 * state_dim]

            # Unflatten z(t) and a(t)
            z_i = z_i.view(batch_size, *state_shape)
            a_i = a_i.view(batch_size, *state_shape)
            
            # Calcculate the vector-jacobian products
            with torch.set_grad_enabled(True):
                # time and state copy vars
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                
                # Calculate f and vjps using the ODEFunction forward_grad method
                func_eval, vjp_f_z, vjp_f_t, vjp_f_params = func.forward_grad(z_i, t_i, grad_out = a_i)
                
                # If any gradient none, set to all 0s
                vjp_f_z = vjp_f_z.to(z_i.device) if vjp_f_z is not None else torch.zeros(batch_size, *state_shape).to(z_i.device)
                vjp_f_t = vjp_f_t.to(z_i.device) if vjp_f_t is not None else torch.zeros(batch_size, 1).to(z_i.device)
                vjp_f_params = vjp_f_params.to(z_i.device) if vjp_f_params is not None else torch.zeros(batch_size, num_params).to(z_i.device)

            # Flatten the output and function grad wrp to z
            func_eval = func_eval.view(batch_size, state_dim)
            vjp_f_z = vjp_f_z.view(batch_size, state_dim)
            
            # Return the augmented state: (f, -a df/dz, -a df/dp, -a df/dt)
            return torch.cat((func_eval, -vjp_f_z, -vjp_f_params, -vjp_f_t), dim = 1)

        # Flatten grad_out
        grad_out = grad_out.view(num_timesteps, batch_size, state_dim)
        
        with torch.no_grad():
            # Will hold calculated gradients for state and params
            adj_z = torch.zeros(batch_size, state_dim).to(grad_out.device)
            adj_p = torch.zeros(batch_size, num_params).to(grad_out.device)
            
            # Also need all gradients across time
            adj_t = torch.zeros(num_timesteps, batch_size, 1).to(grad_out.device)

            # Next, move back in time use ODE solver and vjp's to find the gradients
            for i in range(num_timesteps - 1, 0, -1):
                # Get the current state, time, and functionoutput
                z_current = z[i]
                t_current = t[i]
                f_current = func(z_current, t_current).view(batch_size, state_dim)

                # Compute grads
                direct_grad_z = grad_out[i]
                direct_grad_t = torch.bmm(direct_grad_z.unsqueeze(-2), f_current.unsqueeze(-1))[:, 0]

                # Update adjoints
                adj_z += direct_grad_z
                adj_t[i] = adj_t[i] - direct_grad_t

                # Create the augmented variable
                aug_z = torch.cat((z_current.view(batch_size, state_dim), adj_z, torch.zeros(batch_size, num_params).to(z.device), adj_t[i]), dim = -1)

                # Solve ODE backwards
                aug_ans = ode_solver(aug_z, t_current, t[i - 1], aug_dynamics)

                # Unpack the augmented state (solved backwards)
                adj_z[:] = aug_ans[:, state_dim: 2 * state_dim]
                adj_p[:] += aug_ans[:, 2 * state_dim: 2 * state_dim + num_params]
                if i > 0:
                    adj_t[i - 1] = aug_ans[:, 2 * state_dim + num_params:]

                # Delete vars for memory efficiency
                del aug_z, aug_ans

            # Finally, adjust 0 time adjoint
            direct_grad_z_0 = grad_out[0]
            direct_grad_t_0 = torch.bmm(direct_grad_z_0.unsqueeze(-2), f_current.unsqueeze(-1))[:, 0]
            adj_z += direct_grad_z_0
            adj_t[0] = adj_t[0] - direct_grad_t_0
            
        # Return the gradients wrp state, time, and params
        return adj_z.view(batch_size, *state_shape), adj_t, adj_p, None, None
    
# Implements given function as a Neural Net (Neural ODE)
class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    # Forward pass just uses forward method
    def forward(self, z_init, t, save_all = False, ode_solver = None):
        t = t.to(z_init.device)
        z = ODEAdjoint.apply(z_init, t, self.func.flatten_parameters(), self.func, ode_solver)
        if save_all:
            return z
        else:
            return z[-1]

# For plotting the solutions to ODEs
def plot_ODE_sol(observations = None, times = None, pred_path = None, figname = None):
        plt.figure(figsize = (16, 8))
        # First plot the actual points
        if observations is not None:
            if times is None:
                times = [None] * len(observations)
            for observation, time in zip(observations, times):
                observation, time = observation.detach().cpu().numpy(), time
                # Iterate over all batch points, plot them
                for b_i in range(observation.shape[1]):
                    plt.scatter(observation[:, b_i, 0], observation[:, b_i, 1])
        # If the prediction is available, also plot it
        if pred_path is not None:
            for pred_point in pred_path:
                pred_point = pred_point.detach().cpu().numpy()
                plt.plot(pred_point[:, 0, 0], pred_point[:, 0, 1], c = 'black')
        # Save the figure if name available 
        if figname is not None:
            plt.savefig(figname)
        plt.close()
        
# For generating batches 
def create_batch(observation_data, time_data_np, time_tensor, t_max_val, index_np, num_points):
    
        min_delta_time = 1.0
        max_delta_time = 5.0
        t0 = np.random.uniform(0, t_max_val - max_delta_time)
        t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

        # Get random indices for specified time range
        idx = sorted(np.random.permutation(index_np[(time_data_np > t0) & (time_data_np < t1)])[:num_points])
        
        # Generate the batch
        observation_batch = observation_data[idx]
        time_batch = time_tensor[idx]
        return observation_batch, time_batch
