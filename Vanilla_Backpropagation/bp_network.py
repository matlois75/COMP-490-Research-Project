import torch.nn as nn

class BPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, output_dim: int):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return logits