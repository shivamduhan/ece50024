import torch
from torch.autograd import Variable

spiral_matrix = torch.Tensor([[-0.1, 2.], [-2., -0.1]]) # Matrix that describes the generated spiral
z_init = Variable(torch.Tensor([[-4.0, -2.0]]))         # Initial starting point for the spiral
