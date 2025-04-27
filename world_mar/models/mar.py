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
                 img_height=360, img_width=640, 
                 patch_size=2, token_embed_dim=16,
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
        self.token_embed_dim = token_embed_dim * patch_size**2
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True) # projs VAE latents to transformer dim
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, qkv_bias=True,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        # TODO: embs for ctx, prev, action

        # --- decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, qkv_bias=True,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # TODO: embs for ctx, prev, action

        # --- pose prediction ---
        # TODO: add pose prediction network, throw into training

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
        assert self.vae.latent_dim == token_embed_dim
        self.seq_h, self.seq_w = self.vae.seq_h // patch_size, self.vae.seq_w // patch_size
        self.frame_seq_len = self.seq_h, self.seq_w
        # we assume here the diffusion model operates one frame at a time:
        self.diffusion_pos_emb_learned = nn.Parameter(torch.zeros(1, self.frame_seq_len, decoder_embed_dim))
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_emb_learned, std=.02)

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
    
    def patchify(self, x):
        bsz, s, c = x.shape
        p = self.patch_size
        h_, w_ = self.seq_h // p, self.seq_w // p
        x = rearrange(x, "b s c -> b c h w", h=self.seq_h)

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]
    
    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.frame_seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_gen.rvs(1)[0]
        num_masked_tokens = int(np.ceil((self.frame_seq_len) * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # TODO: consider moving this offset to any frame?
        #       this is 0 currently because pred frame is at start of seq
        offsets = torch.zeros(bsz, 1, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=offsets + orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask, offsets

    def forward_encoder(self, x, actions, poses, mask):
        # x : expected to be b (t s) d
        x = self.z_proj(x)
        bsz, _, embed_dim = x.shape

        # TODO: add embs based on pos (RoPE), actions, poses, want:
        #       x_i + E_i, Ei = E_ai + E_pi
        x = ...
        x = self.z_proj_ln(x)

        x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x
    
    def forward_decoder(self, x, actions, poses, mask):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1).to(x.dtype) # creates b (t s) d, all mask token
        x_full = mask_tokens.clone()
        x_full[(1-mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_full

        # TODO: add embs based on pos (RoPE), actions, poses, want:
        #       x_i + E_i, Ei = E_ai + E_pi
        # THESE SHOULD BE DIFF FROM THE ENCODER ONES
        x = ...
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
    
    def forward_diffusion(self, z, tgt, mask):
        tgt = rearrange(tgt, "b s d -> (b s) d").repeat(self.diffusion_batch_mul, 1)
        z = rearrange(z, "b s d -> (b s) d").repeat(self.diffusion_batch_mul, 1) # ad
        mask = rearrange(mask, "b s -> (b s)").repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=tgt, mask=mask)
        return loss

    def forward(self, frames, actions, poses):
        # TODO: fill this out more detail

        b, t, c, h, w = frames.shape

        # 1) compress frames w/ vae
        x = rearrange(frames, "b t c h w -> (b t) c h w")
        x = self.vae.encode(frames).sample() # (b t) (seq_h seq_w) z_dim
        x = self.patchify(x)
        x = rearrange(x, "(b t) s d -> b (t s) d", b=b)
        x_gt = x.clone().detach()

        # 2) gen mask
        orders = self.sample_orders(b)
        mask, offsets = self.random_masking(x, orders)

        # 2) run encoder
        x = self.forward_encoder(x, actions, poses, mask)

        # 3) run decoder
        z = self.forward_decoder(x, actions, poses, mask)

        # 4) split into tgt frame + diffuse
        idx = offsets + torch.arange(self.frame_seq_len)
        idx = idx.unsqueeze(-1).expand(-1,-1,z.shape[-1])
        z_t = torch.gather(z, dim=1, index=idx) 
        xt_gt = torch.gather(x_gt, dim=1, index=idx)
        # WARNING: THIS IS BAD IF WE ARE DOING DIFF FRAME PRED!!!!
        mask_t = mask[:,:self.frame_seq_len]

        loss = self.forward_diffusion(z_t, xt_gt, mask_t)        

        return loss

    def training_step(self, batch, batch_idx):
        # TODO: parse batch, whether dict or tuple (MOVE TO DEVICE)
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # parse batch 
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, C, H, W]
        actions = batch["actions"].to(self.device) # shape ...
        poses = batch["poses"].to(self.device) # shape [B, T, 5]

        loss = self(frames, actions, poses)
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
