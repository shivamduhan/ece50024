import torch
from torch.autograd import Variable

spiral_matrix = torch.Tensor([[-0.1, -1.], [1., -0.1]]) # Matrix that describes the generated spiral
z_init = Variable(torch.Tensor([[0.6, 0.3]]))           # Initial starting point for the spiral
