import torch
from torch import nn


class LinearProjection(nn.Module):
    def __init__(self, ac: int, channels: int, patch_num: int, d_model: int = 248, bias:bool = True):
        super(LinearProjection, self).__init__()
        self.dct = ac + 1
        self.channels = channels
        self.d_model = d_model
        self.patch_num = patch_num


        self.projection = nn.Linear(self.dct * channels, d_model, bias=bias)

    def init_weights(self, init_fn):
        init_fn(self.projection.weight)

    def forward(self, X):
        batch_size = X.shape[:-3]
        permutate_dim = (0, 2, 1, 3) if batch_size else (1, 0, 2)

        X = X.permute(permutate_dim).reshape(*batch_size, self.patch_num, -1)
        return self.projection(X)



class LinearProjectionOfFlattenedPatches(nn.Module):
    def __init__(self, in_channels: int, heigth: int, width: int, patch_size: int, d_model: int):
        super(LinearProjectionOfFlattenedPatches, self).__init__()
        self.img_size = (heigth, width)
        self.patch_size = patch_size  # P
        self.in_channels = in_channels  # C

        self.N = (self.img_size[0] * self.img_size[1]) / (patch_size**2)
        assert self.N % 1 == 0, f"num_patches must be divisible by patch size: {patch_size}, size: {self.img_size}"
        self.N = int(self.N)
        patch_dimensions = patch_size * patch_size * in_channels

        # Linear projection of patches into latent vector of dimension d_model
        self.projection = nn.Linear(patch_dimensions, d_model)

    def forward(self, x):
        batch_size = x.shape[:-3]
        permutate_dim = (0, 2, 1, 3) if batch_size else (1, 0, 2)

        # Divide images into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(*batch_size, self.in_channels, self.N, -1)
        x = torch.permute(x, permutate_dim).contiguous().view(*batch_size, self.N, -1)

        # Apply linear projection to each patch
        x = self.projection(x)

        return x