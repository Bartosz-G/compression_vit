import torch
from torch import nn
import torch.nn.functional as F
from nn import LinearProjection

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def init_weights(self, init_fn):
        init_fn(self.conv1)
        init_fn(self.conv2)
        if self.downsample is not None:
            for m in self.downsample:
                init_fn(m)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, channels:int, num_classes: int):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        # Purposely removed nn.MaxPool2d to prevent from downsampling too much on a 32x32 picture
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
                     ),
            ResBlock(128, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
                     ),
            ResBlock(256, 256, stride=1)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=(1, 1), stride=2, bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
                     ),
            ResBlock(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def init_weights(self, init_fn):
        init_fn(self.conv1)
        for m in self.layer1:
            m.init_weights(init_fn)

        for m in self.layer2:
            m.init_weights(init_fn)

        for m in self.layer3:
            m.init_weights(init_fn)

        for m in self.layer4:
            m.init_weights(init_fn)

        init_fn(self.fc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x



class NeuralNet(nn.Module):
    def __init__(self,
                 ac: int,
                 channels: int,
                 patch_num: int,
                 num_classes: int,
                 hidden_size: int = 4,
                 dim_feedforward: int = 1024,
                 activation = nn.ReLU(),
                 bias = True,) -> None:
        super(NeuralNet, self).__init__()
        self.dct = ac + 1

        self.linear_projection = LinearProjection(ac=ac,
                                                  channels=channels,
                                                  patch_num=patch_num,
                                                  d_model=dim_feedforward,
                                                  bias=bias)

        layer_list = nn.ModuleList()
        for _ in range(hidden_size):
            layer_list.append(
                nn.Linear(dim_feedforward, dim_feedforward, bias=bias),
                nn.BatchNorm1d(dim_feedforward),
                activation,
            )


        self.backbone = nn.Sequential(*layer_list)
        self.head = nn.Linear(dim_feedforward, num_classes, bias=bias)

    def init_weights(self, init_fn) -> None:
        init_fn(self.linear_projection.weight)
        init_fn(self.head.weight)
        init_fn(self.backbone.weight)

    def forward(self, x):
        x = self.linear_projection(x)
        x = self.backbone(x)
        return self.head(x)




