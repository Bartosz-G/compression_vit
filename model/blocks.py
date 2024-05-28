import torch
from torch import nn
import warnings


class LinearProjection(nn.Module):
    def __init__(self, ac: int, channels: int, patch_num: int, d_model: int = 248, bias:bool = True):
        super(LinearProjection, self).__init__()
        warnings.warn(f'LinearProjection is depretiated since FlattenZigZag() from transforms correctly adjusts the dimensions')
        self.dct = ac + 1
        self.channels = channels
        self.d_model = d_model
        self.patch_num = patch_num


        self.projection = nn.Linear(self.dct * channels, d_model, bias=bias)

    def init_weights(self, init_fn):
        init_fn(self.projection.weight)

    def forward(self, X): # Expects input [batch_size, channels, patch_num, pixels_in_patch]
        batch_size = X.shape[:-3]
        permutate_dim = (0, 2, 1, 3) if batch_size else (1, 0, 2)

        X = X.permute(permutate_dim).reshape(*batch_size, self.patch_num, -1)
        return self.projection(X)