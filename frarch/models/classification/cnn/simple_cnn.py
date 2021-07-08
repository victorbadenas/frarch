import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, kernels, channels, dilation=None, paddings=None, strides=None):
        super(SimpleCNN, self).__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(make_block(i))

    def make_block(self, i, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d()
        )
