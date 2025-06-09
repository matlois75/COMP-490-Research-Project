import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_layer(layer):
    """
    Initializes the weights and biases of layers.py
    """
    if not isinstance(layer, nn.Linear):
        return
    
    init.orthogonal_(layer.weight)
    layer.bias.data.zero_()
    
def initialize_network(model):
    for module in model.modules():
        if (isinstance(module, nn.Linear)):
            initialize_layer(module)

def add_gaussian_noise(tensor, sigma):
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise