"""
Module: residual_encoder.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import torch.nn as nn

from models.utils import get_padding
from models.residual_base import ResidualBase, get_block_sizes, get_block_type

DefaultEncoderOpts = {
    "input_channels": 1,
    "conv_kernel_size": 7,
    "max_pool": 1,
    "resnet_size": 18,
}

"""
Defines the convolutional or feature extraction part (residual layers) of the CNN. According to the chosen and supported 
ResNet architecture (ResNet18, 34, 50, 101, 152) the block types and sizes are generated in order to construct the 
respective residual encoder part as well as the forward path computation.
"""
class ResidualEncoder(ResidualBase):
    def __init__(self, opts: dict = DefaultEncoderOpts):
        super().__init__()
        self._opts = opts
        self.cur_in_ch = 64
        self.block_sizes = get_block_sizes(opts["resnet_size"])
        self.block_type = get_block_type(opts["resnet_size"])

        self.conv1 = nn.Conv2d(
            opts["input_channels"],
            out_channels=64,
            kernel_size=opts["conv_kernel_size"],
            stride=(2, 2),
            padding=get_padding(opts["conv_kernel_size"]),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.cur_in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        if opts["max_pool"] == 1:
            self.max_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=get_padding(3)
            )
            stride1 = (1, 1)
        elif opts["max_pool"] == 0:
            self.max_pool = None
            stride1 = (2, 2)
        elif opts["max_pool"] == 2:
            self.max_pool = None
            stride1 = (1, 1)

        self.layer1 = self.make_layer(self.block_type, 64, self.block_sizes[0], stride1)
        self.layer2 = self.make_layer(self.block_type, 128, self.block_sizes[1], (2, 2))
        self.layer3 = self.make_layer(self.block_type, 256, self.block_sizes[2], (2, 2))
        self.layer4 = self.make_layer(self.block_type, 512, self.block_sizes[3], (2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def model_opts(self):
        return self._opts
