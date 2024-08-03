# inspired by https://nn.labml.ai/diffusion/ddpm/unet.html

import math
from typing import Tuple, Union, List

import torch
from torch import nn
import copy

"""

Implements the VQ-VAE model used for joint training of the encoder and decoder modules

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
        self.conv = nn.ConvTranspose3d(n_channels, n_channels, (4, 4, 4), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv3d(n_channels, n_channels, (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim  
        self._num_embeddings = num_embeddings 
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):        
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()   
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = torch.nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

class VQVAE(nn.Module):

    def __init__(
        self,
        n_elements: int = 4,
        n_latent: int = 4,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0.1,
        smooth_sigma: float = 0.0,
        skip_connections: int = 1,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.smooth_sigma = smooth_sigma
        self.skip_connections = skip_connections
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        n_resolutions = len(ch_mults)

        self.grid_projection = nn.Conv3d(n_elements, n_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

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

        # reduce number of channels
        self.conv_out = nn.Conv3d(out_channels, n_latent, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # quantize
        self._pre_vq_conv = nn.Conv3d(n_latent, self.embedding_dim, 
                                      kernel_size=(1,1,1), stride=1).to("cuda:0")
        
        self._vq_vae = VectorQuantizer(self.num_embeddings, self.embedding_dim,
                                           self.commitment_cost)

        # decoder
        # increase the number of latent chanels 
        self.conv_out_transpose = nn.ConvTranspose3d(self.embedding_dim, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
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
        self.final = nn.Conv3d(in_channels, n_elements, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.grid_projection(x)

        # encoder
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x) 
            
        # bottleneck
        x = self.middle(x)
        x = self.conv_out(x)  

        embeddings = copy.deepcopy(x.detach())
        self.avg_pool = nn.AvgPool3d(kernel_size=4, stride=None)  
        embeddings = torch.squeeze(self.avg_pool(embeddings))

        x = self._pre_vq_conv(x)
        loss, quantized, perplexity, _ = self._vq_vae(x) 

        # decoder
        x = self.conv_out_transpose(quantized) 
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

        return loss, x, perplexity   

    def score(self, y):
        xhat, _ = self.forward(y)
        return (xhat - y) / (self.smooth_sigma ** 2)

    