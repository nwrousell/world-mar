"""
Adapted from: https://github.com/LTH14/mar/blob/main/models/mar.py
"""


from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from einops import rearrange

from timm.models.vision_transformer import Block

# from models.diffloss import DiffLoss

from world_mar.modules.utils import instantiate_from_config
from world_mar.oasis_utils.vae import AutoencoderKL
from world_mar.models.diffloss import DiffLoss

import pytorch_lightning as pl


class WorldMAR(pl.LightningModule):
    """
    Assumptions Praccho's making:
        vae is an AutoencoderKL, future me can change this to any encoder decoder, but we like LDM's
        so we should be using it
    
    Req'd args... you need to give me a:
        - vae: thing that has an encode and decode

    """
    def __init__(self, 
                 vae_config, # should be an AutoencoderKL 
                 diffloss_config,
                 img_height=360, img_width=640, num_frames=8,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
                 diffloss_w=512, diffloss_d=3, num_sampling_steps='100', diffusion_batch_mul=4,
                 mask_ratio_min=0.7,
                 proj_dropout=0.1,
                 attn_dropout=0.1,
                 warmup_steps=10000, # TODO: change this depending on dataset size
                 **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False


        # --- masking statistics ---
        # ref: masking ratio used by MAR for image gen
        self.mask_ratio_gen = stats.truncnorm((mask_ratio_min -1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --- encoder ---
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True) # projs VAE latents to transformer dim
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, qkv_bias=True,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # --- decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, qkv_bias=True,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.initialize_weights()
        
        # --- initialize diff loss ---
        # TODO: make cutomizable as MLP (per patch?) vs DiT (per frame).
        #       for now, assuming more lightweight MLP
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        # --- intialize the vae ---
        self.instantiate_vae(vae_config)
        assert isinstance(self.vae, AutoencoderKL)
        self.seq_h, self.seq_w, self.seq_t = self.vae.seq_h, self.vae.seq_w, num_frames
        self.seq_len = self.seq_h * self.seq_w * self.seq_t
        self.token_embed_dim = self.vae.latent_dim
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def instantiate_vae(self, vae_config):
        self.vae = instantiate_from_config(vae_config)
        self.vae.train = lambda self, mode=True: self
        for param in self.vae.parameters():
            param.requires_grad=False
    
    def forward_encoder(self, x, ):
        ...
    
    def forward_decoder(self, ):
        ...

    def forward(self, frames, actions, poses):
        # TODO: fill this out more detail

        B, T, C, H, W = frames.shape

        # 1) encode frames
        x = rearrange(frames, "b t c h w -> (b t) c h w")
        x = self.vae.encode(frames).sample() # (b t) seq_h seq_w z_dim
        x = rearrange(x, "(b t) s d -> b (t s) d", b=B)

        # 2) encode actions and poses


        # 3) add pose embeddings to latents

        # 4)
        z = ...

        # 5) diffuse
        

    def training_step(self, batch, batch_idx):
        # TODO: parse batch, whether dict or tuple (MOVE TO DEVICE)
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # parse batch 
        frames = batch["frames"].to(self.device) # shape [B, T, C, H, W], last two frames are prev frame + pred frame
        actions = batch["actions"].to(self.device) # shape ...
        poses = batch["poses"].to(self.device) # shape [B, T, 5]

        loss = ... # TODO: compute loss
        self.manual_backward(loss)

        opt.step()
        lr_sched.step()
    
    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate)
        lr_sched = LinearLR(optim, start_factor=1.0, end_factor=1.0, total_iters=self.warmup_steps)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "interval": "step",
                "frequency": 1,
            }
        }
