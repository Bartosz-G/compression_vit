from typing import Optional, Callable
from functools import partial
import math

import torch
from torch import nn
import warnings
from torchvision.models.vision_transformer import VisionTransformer as ViT


def activation_getter(activation: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "softmax": nn.Softmax(dim=1),
        "log_softmax": nn.LogSoftmax(dim=1),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
        "softplus": nn.Softplus(),
        "softsign": nn.Softsign(),
        "hardtanh": nn.Hardtanh(),
        "prelu": nn.PReLU(),
        "rrelu": nn.RReLU(),
        "softmin": nn.Softmin(dim=1),
    }

    return activations[activation.casefold()]



class CompressedVisionTransformer(ViT):
    def __init__(self, **kwargs):
        self.dct = kwargs['patch_size'] + 1
        kwargs['patch_size'] = 8 # To supress invalid shape error coming from VisionTransformer base class
        super().__init__(**kwargs)


        self.conv_proj = None
        self.lin_proj = nn.Linear(3 * self.dct, self.hidden_dim)

        fan_in = self.lin_proj.in_features
        nn.init.trunc_normal_(self.lin_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.lin_proj.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, n_hw, c_dct = x.shape
        torch._assert(c_dct == self.dct*3, f"Wrong number of dct values! Expected {self.dct*3} but got {c_dct}!")

        # (n, (n_h * n_w), c*p*p) -> (n, (n_h * n_w), hidden_dim)
        x = self.lin_proj(x)
        # c_dct is equivalent to c*dct which is equivalent to c*p*p for dct == p*p
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        return x




class LinVisionTransformer(ViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_proj = None
        self.lin_proj = nn.Linear(self.patch_size * self.patch_size * 3, self.hidden_dim)
        self.unfold = nn.Unfold(kernel_size=(kwargs['patch_size'], kwargs['patch_size']), stride=(kwargs['patch_size'], kwargs['patch_size']))

        fan_in = self.lin_proj.in_features
        nn.init.trunc_normal_(self.lin_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.lin_proj.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")

        # (n, c, h, w) -> (n, n_h * n_w, p*p*3)
        x = self.unfold(x).transpose(-2, -1)
        # (n, n_h * n_w, p*p*3) -> (n, n_h * n_w, hidden_dim)
        return self.lin_proj(x)

