from typing import List
from itertools import repeat
import collections.abc

from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair

from models.complex_nn import *
from utils.utils import *


class ResidualBlock(nn.Module):
    """
    Pre-Activation ResNet
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()

        self.norm_1 = ComplexBatchNorm2d(in_channels)
        self.act_1 = nn.LeakyReLU(inplace=True)
        self.conv_1 = ComplexConv2d(in_channels, out_channels, kernel_size=(5, 7), padding=(2, 3))

        self.norm_2 = ComplexBatchNorm2d(out_channels)
        self.act_2 = nn.LeakyReLU(inplace=True)
        self.conv_2 = ComplexConv2d(out_channels, out_channels, kernel_size=(5, 7), padding=(2, 3))

        if in_channels != out_channels:
            self.shortcut = ComplexConv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor):
        """
        inputs: [batch, in_channels, h, w]
        t: [batch, time_channels]
        """
        out = self.conv_1(self.act_1(self.norm_1(inputs)))
        out = self.conv_2(self.act_2(self.norm_2(out)))
        return out + self.shortcut(inputs)


#######################################
# Atrous Spatial Pyramid Pooling
#######################################
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [ComplexDepthwiseConv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                   ComplexBatchNorm2d(out_channels),
                   nn.LeakyReLU(inplace=True)]
        super().__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(ComplexAdaptiveAvgPool2d(1),
                                   ComplexDepthwiseConv2d(in_channels, out_channels, 1, bias=False),
                                   ComplexBatchNorm2d(out_channels),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-3:-1]
        out = self.layer(x)
        return ComplexInterpolate(out, size=size)


class ASPP(nn.Module):

    def __init__(self, in_channels: int, atrous_rates=[3, 9], out_channels: int = 256) -> None:
        super().__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(ComplexDepthwiseConv2d((len(self.convs) + 1) * out_channels, out_channels, 1, bias=False),
                                     ComplexBatchNorm2d(out_channels),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = torch.cat((x, res), dim=1)
        return self.project(res)
