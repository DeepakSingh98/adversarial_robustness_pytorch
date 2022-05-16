
import torch.nn as nn
import torch

class NormOut(nn.Module):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer across the batch. 
    """
    def forward(self, x):
        if self.training:
            x_prime = abs(x)
            if len(x.shape) == 2: # x is batch x features
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0] # across batch
                x_prime_max = torch.max(x_prime_max, dim=1, keepdim=True)[0] # across layer
            elif len(x.shape) == 4: # x is batch x channels x height x width
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0] # across batch
                x_prime_max = torch.max(x_prime_max, dim=2, keepdim=True)[0] # across height
                x_prime_max = torch.max(x_prime_max, dim=3, keepdim=True)[0] # across width
            else:
                raise ValueError("Input dimension not supported")
            norm_x = x_prime / x_prime_max
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        return x
