import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        # weight init happens in utils.py

    def forward(self, x):
        """This module computes out = tanh(Wx + b)

        self.linear.weight (W) must have shape (out_dim, in_dim)

        DTP paper uses tanh activation function presumably for its symmetry about 0 encouraging nicer inverses.

        Args:
            x (_type_): batched input tensor with shape (batch_size, in_dim)

        Returns:
            _type_: must have shape (batch_size, out_dim)
        """
        h = self.linear(x)
        return torch.tanh(h)

class InverseLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_inv = nn.Linear(out_dim, in_dim)
        # weight init happens in utils.py

    def forward(self, h):
        """This module computes recon = tanh(W_g h + b_g)

        self.linear_inv.weight (W_g) must have shape (in_dim, out_dim)

        Args:
            h (_type_): batched input tensor; post-activation output from forward layer with shape (batch_size, out_dim)

        Returns:
            _type_: must have shape (batch_size, in_dim)
        """
        recon = self.linear_inv(h)
        return torch.tanh(recon)