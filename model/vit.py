import torch
from torch import nn
from nn import LinearProjection, LinearProjectionOfFlattenedPatches




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




class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 height: int,
                 width: int,
                 patch_size: int,
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
        super(VisionTransformer, self).__init__()
        self.img_size = (height, width)

        # Calculating the correct dimensions:
        N = (self.img_size[0] * self.img_size[1]) / (patch_size**2)
        assert N % 1 == 0, f"num_patches must be divisible by patch size: {patch_size}, size: {self.img_size}"
        N = int(N)

        # [class] token
        self.class_embedding = nn.Parameter(torch.randn(1, d_model))

        self.position_embedding = nn.Parameter(torch.randn(N + 1, d_model)) if learnable_positional else None


        self.linear_projection_of_flattened_patches = LinearProjectionOfFlattenedPatches(
            in_channels, height, width, patch_size, d_model)

        transformer_encoder_list = nn.ModuleList()
        for _ in range(ntransformers):
            transformer_encoder_list.append(torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=nn.ReLU(),
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,))

        self.transformer_encoder = nn.Sequential(*transformer_encoder_list)

        self._pre_training_head = nn.Sequential(nn.LayerNorm(d_model, eps=layer_norm_eps),
                                                nn.Linear(d_model, dim_feedforward),
                                                activation,
                                                nn.Linear(dim_feedforward, num_classes))

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

    def _with_positional(self, X):
        return X + self.positional if self.learnable_positional else X

    def _concat_cls_token(self, X):
        batch_size = X.shape[:-2]
        if batch_size:
            cls_token = self.cls_token.expand(*batch_size, -1, -1)
            return torch.cat((cls_token, X), dim=1)
        cls_token = self.cls_token.squeeze(0)
        return torch.cat((cls_token, X), dim = 0)

    def forward(self, X):
        batch_size = X.shape[:-3]
        patches = self.linear_projection_of_flattened_patches(X)
        patches = self._concat_cls_token(patches)
        patches = self._with_positional(patches)
        patches = self.transformer_encoder(patches)
        class_token = patches[:,0,:] if batch_size else patches[0, :]
        output = self.head(class_token)
        return output







