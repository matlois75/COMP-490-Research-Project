import torch
import torch.nn as nn

class BPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.layers.append(nn.Tanh())
            current_dim = h_dim
        
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)
        return logits