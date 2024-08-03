# inspired by https://nn.labml.ai/diffusion/ddpm/unet.html

import math
from typing import Tuple, Union, List

import torch
from torch import nn
import copy
import numpy as np

"""

Implements the VQ-VAE decoder class used to decode latent embeddings back to 3D voxels

"""
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_groups: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.use_norm = n_groups > 0
        # first norm + conv layer
        if self.use_norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        # second norm + conv layer
        if self.use_norm:
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.shortcut = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        if self.use_norm:
            h = self.norm1(x)
            h = self.act1(h)
        else:
            h = self.act1(x)
        h = self.conv1(h)

        if self.use_norm:
            h = self.norm2(h)
        h = self.act2(h)
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 16
    ):

        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.use_norm = n_groups > 0
        self.n_heads = n_heads
        self.d_k = d_k

        if self.use_norm:
            self.norm = nn.GroupNorm(n_groups, n_channels)

        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)

        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = d_k ** -0.5

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width, depth = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum("bihd, bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)

        res = torch.einsum("bijh, bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        
        res = self.output(res)
        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width, depth)
        return res
    

class UpBlock(nn.Module):
    """
    This combines ResidualBlock and AttentionBlock.
    These are used in the second half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float, skip_connections: int):
        super().__init__()
        if skip_connections:
            self.res = ResidualBlock(in_channels + out_channels, out_channels, n_groups=n_groups, dropout=dropout)
        # if no skip connections
        else:
            self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Decoder3D_VQVAE(nn.Module):

    def __init__(
        self,
        n_elements: int = 4,
        n_latent: int =4,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        skip_connections: int = 1,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.skip_connections = skip_connections
        self.embedding_dim = embedding_dim
        n_resolutions = len(ch_mults)

        # decoder   
        out_channels = n_channels * np.prod(ch_mults)  
        in_channels = out_channels

        # increase number of channels 
        self.conv_out_transpose = nn.ConvTranspose3d(self.embedding_dim, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        up = []
        
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout, skip_connections))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_groups, is_attn[i], dropout, skip_connections))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        if n_groups > 0:
            self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv3d(in_channels, n_elements, kernel_size=(3, 3, 3), padding=(1, 1, 1))


    def forward(self, x: torch.Tensor, hh):
        # undo conv_out
        x = self.conv_out_transpose(x)  
        # decoder
        for m in self.up:
            if isinstance(m, Upsample) or self.skip_connections==0:  # no skip connections
                x = m(x)
            else:
                ss = hh.pop()  
                x = torch.cat((x, ss), dim=1)
                x = m(x)
        
        if hasattr(self, "norm"):
            x = self.norm(x)  
        x = self.act(x)
        x = self.final(x)
        return x