import torch
from torch import nn

from svtr import Block, ConvBNLayer


class SVTREncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        dims=64,  # XS
        depth=2,
        hidden_dims=120,
        use_guide=True,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.,
        qk_scale=None
    ):
        super(SVTREncoder, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvBNLayer(in_channels, in_channels // 8, padding=1, act=nn.SiLU)
        self.conv2 = ConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act=nn.SiLU)

        self.svtr_block = nn.ModuleList(
            [
                Block(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer='Global',
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer='nn.LayerNorm',
                    epsilon=1e-05,
                    prenorm=False
                ) for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=nn.SiLU)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvBNLayer(2 * in_channels, in_channels // 8, padding=1, act=nn.SiLU)

        self.conv1x1 = ConvBNLayer(in_channels // 8, dims, kernel_size=1, act=nn.SiLU)
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # for use guide
        if self.use_guide:
            z = x.clone()
            z.stop_gradient = True
        else:
            z = x

        # for short cut
        h = z

        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)

        # SVTR global block
        B, C, H, W = z.shape
        z = torch.squeeze(z, dim=2)
        z = torch.transpose(z, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)

        # last stage
        z = z.reshape([x.shape[0], H, W, C])
        z = z.view(z.shape[0], z.shape[3], z.shape[1], z.shape[2])
        z = self.conv3(z)
        z = torch.concat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))

        return z