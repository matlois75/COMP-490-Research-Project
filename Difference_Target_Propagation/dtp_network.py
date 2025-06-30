import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from Difference_Target_Propagation.dtp_layers import ForwardLayer, InverseLayer
from Difference_Target_Propagation.dtp_utils import initialize_network, add_gaussian_noise

class DTPNetwork(nn.Module):
    """Implements a feedforward neural network of L hidden layers using forward layers (f_1, ..., f_L) and corresponding inverse layers (g_1, ..., g_L). Supports Difference Target Propagation training.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple,
            output_dim: int,
            eta_hat: float,
            sigma: float
            ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.eta_hat = eta_hat
        self.sigma = sigma
        
        self.f_layers = nn.ModuleList()
        self.g_layers = nn.ModuleList()
        
        self.f_layers.append(ForwardLayer(input_dim, hidden_dims[0]))
        self.g_layers.append(InverseLayer(input_dim, hidden_dims[0]))
        for i in range(0, len(hidden_dims)-1):
            self.f_layers.append(ForwardLayer(hidden_dims[i], hidden_dims[i+1]))
            self.g_layers.append(InverseLayer(hidden_dims[i], hidden_dims[i+1]))
        self.f_layers.append(ForwardLayer(hidden_dims[-1], output_dim, use_tanh=False))
        # self.g_layers.append(InverseLayer(hidden_dims[-1], output_dim)) # problematic line. excluding it yields good results so may not be needed
        
        initialize_network(self)
        
    def forward(self, x):
        """Standard forward pass of the network using input x and returning the logits and list of hidden activations.

        Args:
            x (tensor): shape (batch_size, input_dim). input_dim flattened if necessary.

        Returns:
            tensor: logits (Tensor of shape (batch_size, output_dim)), and h_list (Python list of length L, where h_list[i] is the i-th hidden activation).
        """
        h_i = x
        h_list = []
        
        # h_0 will be assigned the output of feeding x into the first hidden layer, h_1 will be assigned the output of feeding h_0 into the second hidden layer, and so on.
        for i in range(len(self.f_layers)-1):
            h_i = self.f_layers[i](h_i)
            h_list.append(h_i)
        
        logits = self.f_layers[-1](h_i)
        
        return logits, h_list
    
    def compute_targets(self, h_list, labels, logits):
        """_summary_

        Args:
            h_list (list): the list of hidden activations returned by forward(...). i.e. [h_1, h_2, ..., h_L]
            labels (tensor): the ground-truth class indices to compute classification loss at the top.
            logits (tensor): raw, unnormalized output scores from final linear layer of the model, before any activation function is applied
            
        Returns:
            hhat_list (list): _description_
            loss_top (_type_): _description_
        """
        h_L = h_list[-1]
        loss_top = CrossEntropyLoss()(logits, labels) # computes classification loss
        
        grad_h_L = torch.autograd.grad(
            outputs=loss_top,
            inputs=h_L, # gradient calculations stop at top hidden layer and doesn't propogate further back through earlier layers
            retain_graph=True
        )[0] # produces gradient w.r.t. h_L
        
        hL_target = h_L - (grad_h_L * self.eta_hat) # eta_hat is step size

        hhat_list = [hL_target]

        for i in range(2, len(self.hidden_dims)+1): # TODO: look at/fix equation
            h_i = h_list[-i]
            h_i_plus = h_list[-i+1]
            hhat_i_plus = hhat_list[0]
            g_i_plus = self.g_layers[-i+1]
            
            with torch.no_grad():
                delta = g_i_plus(hhat_i_plus.detach()) - g_i_plus(h_i_plus.detach())
            hhat_list.insert(0, (h_i + delta).detach())
            
        # now we have hhat_list = [hhat_1, hhat_2, ..., hhat_L]
        
        return hhat_list, loss_top
    
    def layer_losses(self, h_list, hhat_list, x):
        """Computes forward reconstruction loss and inverse reconstruction loss.

        Args:
            h_list (list): list of original hidden activations plus input x prepended [x, h_1, h_2, ..., h_L]
            hhat_list (list): list of targets for each hidden layer [hhat_1, hhat_2, ..., hhat_L]
            x (tensor): input sequence to be assigned to the first h_prev

        Returns:
            tuple: Returns two scalars: total_forward_loss = sum of all ||f_i(h_{i-1}) - hhat_i||^2, and total_inverse_loss = sum of all ||g_i(f_i(h_{i-1} + epsilon)) - (h_{i-1} + epsilon)||^2.
        """
        mse_loss = MSELoss(reduction='mean')
        total_forward_loss = torch.tensor(0.0, device=h_list[0].device)
        total_inverse_loss = torch.tensor(0.0, device=h_list[0].device)
        # could also store each layer's individual losses in Python lists to inspect them or weight them differently
        
        for i, hhat_i in enumerate(hhat_list):
            h_prev = x if i==0 else h_list[i-1]
            hhat_i = hhat_list[i]
            forward_i = self.f_layers[i]
            inverse_i = self.g_layers[i]
            
            tilde_h_i = forward_i(h_prev)
            total_forward_loss += mse_loss(tilde_h_i, hhat_i)
            
            noisy = add_gaussian_noise(h_prev, self.sigma) # sigma determines the standard deviation of the Gaussian noise that gets added to h_prev
            f_noisy = forward_i(noisy)
            g_recon = inverse_i(f_noisy)
            total_inverse_loss += mse_loss(g_recon, noisy)
        # now summed up one forward and one inverse loss for each hidden layer. total losses are ready
        
        return (total_forward_loss, total_inverse_loss)
    
    def step(self, x, labels, optimizers, update_forward=True):
        """_summary_

        Args:
            x (tensor): input batch, tensor of shape (batch_size, input_dim)
            labels (tensor): ground truth class indices, tensor of shape (batch_size, 1)
            optimizer (_type_): an optimizer instance created outside like RMSProp
        """
        g_opt = optimizers if not isinstance(optimizers, tuple) else optimizers[1]
        
        # train g
        g_opt.zero_grad()
        logits, h_list = self.forward(x)
        hhat_list, _ = self.compute_targets(h_list, labels, logits)
        _, g_loss = self.layer_losses(h_list, hhat_list, x)
        g_loss.backward()
        g_opt.step()
        
        if not update_forward:
            return 0.0, 0.0, g_loss.item(), 0
        
        # train f
        f_opt = optimizers[0]
        f_opt.zero_grad()
        logits, h_list = self.forward(x)
        hhat_list, loss_top = self.compute_targets(h_list, labels, logits)
        f_loss, _ = self.layer_losses(h_list, hhat_list, x)
        (loss_top + f_loss).backward()
        f_opt.step()
        
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        
        return (loss_top.item(), f_loss.item(), g_loss.item(), correct)