"""
Adapted from: https://github.com/xizaoqu/WorldMem/blob/main/algorithms/worldmem/models/attention.py
"""

from einops import rearrange
from embeddings.rotary_embedding import RotaryEmbedding, apply_rotary_emb
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

class MaskedAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        add_mask: Optional[torch.Tensor] = None
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            add_mask = (~attn_mask).to(q.dtype) * torch.finfo(q.dtype).min
            add_mask = add_mask.view(B, 1, 1, N) 

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=add_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if add_mask:
                attn = attn + add_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskedBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

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

        # --- temporal attention block ---
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