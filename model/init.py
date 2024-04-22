import torch
from torch import nn



def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Initialize a tensor with a truncated normal distribution.
    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        trunc_std (float): Number of standard deviations within which to truncate.
    """
    # Calculate the upper and lower bound.
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < trunc_std) & (tmp > -trunc_std)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

def init_truncated_normal(m):
    """
    Initialize weights with truncated normal distribution and biases to zero.
    """
    if isinstance(m, nn.Linear):
        truncated_normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Parameter):
        # Optional: Initialize only if Parameter has more than 1 dimension (common for weights)
        if m.requires_grad and len(m.shape) > 1:
            truncated_normal_(m.data, mean=0.0, std=0.01)


def init_normal(m):
    """
    Initialize weights with normal distribution and biases to zero for nn.Linear layers.
    Also initialize nn.Parameter with normal distribution if detected.
    """
    if isinstance(m, nn.Linear):
        # Initialize weights with normal distribution
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        # Initialize biases to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Parameter):
        # Check if the parameter requires grad and isn't a bias (optional check)
        if m.requires_grad and len(m.shape) > 1:
            nn.init.normal_(m.data, mean=0.0, std=0.01)