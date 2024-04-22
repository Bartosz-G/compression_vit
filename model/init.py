import torch
from torch import nn
import torch
import numpy as np
import random
from contextlib import contextmanager


@contextmanager
def set_seed(seed):
    state_torch = torch.random.get_rng_state()
    if torch.cuda.is_available():
        state_cuda = torch.cuda.random.get_rng_state()
    state_np = np.random.get_state()
    state_random = random.getstate()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.random.set_rng_state(state_torch)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(state_cuda)
        np.random.set_state(state_np)
        random.setstate(state_random)



def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Initialize a tensor with a truncated normal distribution.
    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        trunc_std (float): Number of standard deviations within which to truncate.
    """

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
        if m.requires_grad and len(m.shape) > 1:
            truncated_normal_(m.data, mean=0.0, std=0.01)


def init_normal(m):
    """
    Initialize weights with normal distribution and biases to zero for nn.Linear layers.
    Also initialize nn.Parameter with normal distribution if detected.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Parameter):
        if m.requires_grad and len(m.shape) > 1:
            nn.init.normal_(m.data, mean=0.0, std=0.01)