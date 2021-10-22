import torch


class BypassModel(torch.nn.Module):
    def __init__(self):
        super(BypassModel, self).__init__()

    def forward(self, x):
        return x
