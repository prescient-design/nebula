# inspired by https://nn.labml.ai/diffusion/ddpm/unet.html

import math
from typing import Tuple, Union, List

import torch
from torch import nn
import copy
import numpy as np
from torch.distributions.uniform import Uniform

"""

Implements latent denoising autoencoder model used for generation of new samples in the latent space

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


class DownBlock(nn.Module):
    """
    This combines ResidualBlock and AttentionBlock .
    These are used in the first half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, has_attn: bool, dropout: float):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, n_groups=n_groups, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=n_groups)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


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


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, n_groups: int, dropout: float):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)
        self.attn = AttentionBlock(n_channels, n_groups=n_groups)
        self.res2 = ResidualBlock(n_channels, n_channels, n_groups=n_groups, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # keep the same dimension
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (5, 5, 5))  
        
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # keep the same dimension
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (5, 5, 5))  

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class UNet3D_Latent_888(nn.Module):

    def __init__(
        self,
        n_latent: int = 4,
        n_channels: int = 64,
        vqvae_out_dim: int = 512,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        skip_connections: int = 1,
    ):
        super().__init__()

        self.smooth_sigma = smooth_sigma
        self.skip_connections = skip_connections
        n_resolutions = len(ch_mults)

        in_channels = vqvae_out_dim  
        
        self.grid_projection = nn.Conv3d(n_latent, n_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # encoder
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_groups, is_attn[i], dropout))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        # bottleneck
        self.middle = MiddleBlock(out_channels, n_groups, dropout)

        # decoder
        up = []
        in_channels = out_channels
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
        self.final = nn.Conv3d(in_channels, n_latent, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        out_channels = vqvae_out_dim # n_channels * np.prod(ch_mults)

    def forward(self, x: torch.Tensor):
        x = self.grid_projection(x)
        # encoder
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x) 

        # bottleneck
        x = self.middle(x)

        # decoder
        for m in self.up:
            if isinstance(m, Upsample) or self.skip_connections==0:  # no skip connections
                x = m(x)
            else:  
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        if hasattr(self, "norm"):
            x = self.norm(x)  
        x = self.act(x)
        x = self.final(x)
        return x 

    def score(self, y):
        xhat = self.forward(y)
        return (xhat - y) / (self.smooth_sigma ** 2)

    @torch.no_grad()
    def wjs_walk_steps(self, y: torch.Tensor, v: torch.Tensor, config: dict):
        """Walk steps of walk-jump sampling.
        Do config["steps_wjs"] Langevin MCMC steps on p(y).
        We Use Sachs et al. discretization of the underdamped Langevin MCMC.
        See the paper and its references on walk jump sampling.

        Args:
            y (torch.Tensor): sample y from mcmc chain
            v (torch.Tensor): _description_
            config (dict): dict with the experiment configurations

        Returns:
            torch.Tensor: y, v
        """

        delta, gamma = config["delta"], config["friction"]
        smooth_sigma = config["smooth_sigma"]
        delta = delta*smooth_sigma
        lipschitz, steps = config["lipschitz"], config["steps_wjs"]
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(-gamma)  # gamma = "effective friction"
        zeta2 = math.exp(-2 * gamma)
        for step in range(steps):
            y += delta * v / 2  # y_{t+1}
            psi = self.score(y)
            v += u * delta * psi / 2  # v_{t+1}
            v = zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * torch.randn_like(y)  # v_{t+1}
            y += delta * v / 2  # y_{t+1}
        torch.cuda.empty_cache()
        return y, v

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor):
        """Jump step of walk-jump sampling.
        Recover clean sample x from noisy sample y.
        It is a simple forward of the network.

        Args:
            y (torch.Tensor): samples y from mcmc chain


        Returns:
            torch.Tensor: estimated ``clean'' samples xhats
        """
        result = self.forward(y) 
        return result

    def initialize_y_v(self, config, mean_lat, std_lat, add_uniform, min_latent, max_latent):
        """initialize (y,v) values for WJS algorithm

        Args:
            config (dict): config of the experiment

        Returns:
            y, v (torch.Tensor): initial values for WJS
        """
        n_channels = len(config["elements"])
        latent_dim = config["embedding_dim"] 
        grid_dim = config["grid_dim_latent"]  
        smooth_sigma = config["smooth_sigma"]
        n_chains = config["n_chains"]

        # gaussian noise
        y = torch.cuda.FloatTensor(n_chains, latent_dim, grid_dim, grid_dim, grid_dim)
        y.normal_(0, smooth_sigma)

        return y, torch.zeros_like(y)
