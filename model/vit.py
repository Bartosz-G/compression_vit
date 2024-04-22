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



class CompressedVisionTransformer(nn.Module):
    def __init__(self,
                 ac: int,
                 channels: int,
                 patch_num: int,
                 num_classes: int,
                 d_model: int = 248,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: int = 0.1,
                 activation = nn.ReLU(),
                 ntransformers: int = 4,
                 layer_norm_eps:float = 1e-5,
                 norm_first: bool = False,
                 bias: bool = True,
                 learnable_positional: bool = True):
        super(CompressedVisionTransformer, self).__init__()

        self.learnable_positional = learnable_positional
        self.activation = activation

        self.linear_projection = LinearProjection(ac=ac,
                                                  channels=channels,
                                                  patch_num=patch_num,
                                                  d_model=d_model,
                                                  bias=bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if learnable_positional:
            self.positional = nn.Parameter(torch.zeros(1, 1 + patch_num, d_model))
        else:
            self.positional = None


        encoders = nn.ModuleList()
        for _ in range(ntransformers):
            encoders.append(
                nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout,
                                           activation=activation,
                                           layer_norm_eps=layer_norm_eps,
                                           batch_first=True,
                                           norm_first=norm_first)
            )

        self.encoder = nn.Sequential(*encoders)

        self._pre_training_head = nn.Sequential(nn.LayerNorm(d_model, eps=layer_norm_eps),
                                                nn.Linear(d_model, dim_feedforward, bias=bias),
                                                activation,
                                                nn.Linear(dim_feedforward, num_classes, bias=bias)
                                                )

        self._fine_tunning_head = nn.Sequential(nn.Linear(d_model, num_classes, bias=bias))

        self.head = self._pre_training_head


    def pre_training(self) -> None:
        self.head = self._pre_training_head


    def fine_tunning(self) -> None:
        self.head = self._fine_tunning_head


    def init_weights(self, init_fn) -> None:
        init_fn(self.cls_token)
        self.linear_projection.init_weights(init_fn=init_fn)
        init_fn(self._fine_tunning_head)
        init_fn(self._pre_training_head)


    def _concat_cls_token(self, X):
        batch_size = X.shape[:-2]
        if batch_size:
            cls_token = self.cls_token.expand(*batch_size, -1, -1)
            return torch.cat((cls_token, X), dim=1)
        cls_token = self.cls_token.squeeze(0)
        return torch.cat((cls_token, X), dim = 0)


    def _with_positional(self, X):
        return X + self.positional if self.learnable_positional else X if X.shape[:-2] else X.unsqueeze(0) # prevents transformer encoder from receiving unbatched input


    def forward(self, X):
        X = self.linear_projection(X)
        X = self._concat_cls_token(X)
        X = self._with_positional(X)
        cls_representation = self.encoder(X)[:, 0, :]
        return self.head(cls_representation)