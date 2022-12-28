"""
Module: residual_base.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import torch.nn as nn

from models.utils import get_padding

"""
Defines a residual basic block unit comprising of two convolutions, batch normalizations, and ReLUs
as well as forward path computation.
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_ch, out_ch, stride=1, downsample=None, upsample=None, mid_ch=None
    ):
        super().__init__()

        if mid_ch is None:
            mid_ch = out_ch

        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")

        if upsample is None:
            self.shortcut = downsample
            self.conv1 = nn.Conv2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                bias=False,
            )
        else:
            self.shortcut = upsample
            self.conv1 = nn.ConvTranspose2d(
                in_ch,
                mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            mid_ch, out_ch, kernel_size=3, stride=1, padding=get_padding(3), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu2(out)

        return out

"""
Defines a residual bottleneck block unit comprising of three convolutions, batch normalizations, and ReLUs
as well as forward path computation.
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_ch, out_ch, stride=1, downsample=None, upsample=None, mid_ch=None
    ):
        super().__init__()

        if mid_ch is None:
            mid_ch = out_ch

        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")
        self.shortcut = None
        if upsample is not None or downsample is not None:
            assert (
                downsample is None or upsample is None
            ), "Only can downsample (encoder) or upsample (decoder) using the shortcut"
            self.shortcut = downsample if downsample is not None else upsample

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)

        if upsample is not None:
            self.conv2 = nn.ConvTranspose2d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
        else:
            self.conv2 = nn.Conv2d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                bias=False,
            )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            in_channels=mid_ch,
            out_channels=out_ch * self.expansion,
            kernel_size=1,
            stride=1,
            padding=get_padding(1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu3(out)

        return out

"""
Makes a residual layer according to the chosen ResNet architecture (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152).
"""
class ResidualBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.cur_in_ch = 64

    def make_layer(self, block, out_ch, size, stride=1, shortcut="downsample"):
        layers = []
        stride_mean = stride
        if isinstance(stride, tuple):
            stride_mean = sum(stride) / len(stride)

        if shortcut == "upsample" and (
            stride_mean > 1 or self.cur_in_ch != out_ch * block.expansion
        ):
            shortcut = nn.ConvTranspose2d(
                in_channels=self.cur_in_ch,
                out_channels=out_ch * block.expansion,
                kernel_size=3,
                stride=stride,
                padding=get_padding(3),
                output_padding=get_padding(stride),
                bias=False,
            )
            layers.append(
                block(
                    in_ch=self.cur_in_ch,
                    out_ch=out_ch,
                    mid_ch=self.cur_in_ch // block.expansion,
                    stride=stride,
                    upsample=shortcut,
                )
            )

        elif shortcut == "downsample" and (
            stride_mean > 1 or self.cur_in_ch != out_ch * block.expansion
        ):
            shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.cur_in_ch,
                    out_channels=out_ch * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
            layers.append(block(self.cur_in_ch, out_ch, stride, downsample=shortcut))
        else:
            layers.append(block(self.cur_in_ch, out_ch))
        self.cur_in_ch = out_ch * block.expansion

        for _ in range(1, size):
            layers.append(block(self.cur_in_ch, out_ch))

        return nn.Sequential(*layers)

"""
Defines block sizes with respect to the various supported ResNet architectures
"""
def get_block_sizes(resnet_size):
    if resnet_size == 18:
        return [2, 2, 2, 2]
    elif resnet_size == 34:
        return [3, 4, 6, 3]
    elif resnet_size == 50:
        return [3, 4, 6, 3]
    elif resnet_size == 101:
        return [3, 4, 23, 3]
    elif resnet_size == 152:
        return [3, 8, 36, 3]
    else:
        raise ValueError("Unsuported resnet size: ", resnet_size)

"""
Defines block type with respect to the various supported ResNet architectures
"""
def get_block_type(resnet_size):
    if resnet_size in [18, 34]:
        return BasicBlock
    elif resnet_size in [50, 101, 152]:
        return Bottleneck
    else:
        raise ValueError("Unsuported resnet size ", resnet_size)
