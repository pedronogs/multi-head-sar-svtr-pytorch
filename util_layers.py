from torch import nn
import torch


class Im2Seq(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == 1
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 2, 1)

        return x
