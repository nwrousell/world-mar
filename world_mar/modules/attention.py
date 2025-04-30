"""
Adapted from: https://github.com/xizaoqu/WorldMem/blob/main/algorithms/worldmem/models/attention.py
"""

from einops import rearrange
from .embeddings.rotary_embedding import RotaryEmbedding, apply_rotary_emb
import numpy as np
from typing import Optional
from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import LayerScale, use_fused_attn
from timm.layers import Mlp, DropPath
from torch.jit import Final
from typing import Optional, Type 

class STBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            spatial_rotary_emb: RotaryEmbedding,
            temporal_rotary_emb: RotaryEmbedding,
            mlp_ratio=4.0,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__()

        approx_gelu = lambda: nn.GELU(approximate="tanh") # fast, they say

        # --- spatial attention block ---
        self.s_norm1 = norm_layer(dim)
        self.spatial_attn = SpatialAxialAttention(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim // num_heads,
            rotary_emb=spatial_rotary_emb 
        )
        self.s_norm2 = norm_layer(dim)
        self.s_mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim*mlp_ratio),
            act_layer=approx_gelu,
            bias=proj_bias,
            drop=proj_drop,
        )

        # --- temporal attention block
        self.t_norm1 = norm_layer(dim)
        self.temporal_attn = TemporalAxialAttention(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim // num_heads,
            rotary_emb=temporal_rotary_emb 
        )
        self.t_norm2 = norm_layer(dim)
        self.t_mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim*mlp_ratio),
            act_layer=approx_gelu,
            bias=proj_bias,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor, s_attn_mask=None, t_attn_mask=None):
        # spatial block
        x = x + self.spatial_attn(self.s_norm1(x), attn_mask=s_attn_mask)
        x = x + self.s_mlp(self.s_norm2(x))

        # temporal block
        x = x + self.temporal_attn(self.t_norm1(x), attn_mask=t_attn_mask)
        x = x + self.t_mlp(self.t_norm2(x))

        return x

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, attn_mask=None):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)

        return x

class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, attn_mask=None):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_mask)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x