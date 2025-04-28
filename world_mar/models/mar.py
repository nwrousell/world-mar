"""
Adapted from: https://github.com/LTH14/mar/blob/main/models/mar.py
"""

from world_mar.modules.embeddings.rotary_embedding import RotaryEmbedding
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
from world_mar.modules.attention import STBlock
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
                 img_height=360, img_width=640, num_frames=5,
                 patch_size=2, token_embed_dim=16,
                 vae_seq_h=32, vae_seq_w=18,
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

        # ----- masking statistics -----
        # ref: masking ratio used by MAR for image gen
        self.mask_ratio_gen = stats.truncnorm((mask_ratio_min -1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # ----- Encoder -----
        # initial projection
        self.token_embed_dim = token_embed_dim * patch_size**2
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True) # projs VAE latents to transformer dim
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        # special tokens [PRED], [PREV], and [CTX]
        self.pred_token = nn.Parameter(torch.zeros(1, vae_seq_w, encoder_embed_dim))
        self.prev_token = nn.Parameter(torch.zeros(1, vae_seq_w, encoder_embed_dim))
        self.ctx_token = nn.Parameter(torch.zeros(1, vae_seq_w, encoder_embed_dim))
        # RoPE rotary embeddings for encoder blocks along spatial and temporal dimensions
        self.enc_spatial_rotary_emb = RotaryEmbedding(dim=encoder_embed_dim // encoder_num_heads // 2, freqs_for="pixel", max_freq=256)
        self.enc_temporal_rotary_emb = RotaryEmbedding(dim=encoder_embed_dim // encoder_num_heads)
        # encoder spatio-temporal attention blocks
        self.encoder_blocks = nn.ModuleList([
            STBlock(
                encoder_embed_dim, encoder_num_heads, qkv_bias=True,
                proj_drop=proj_dropout, attn_drop=attn_dropout,
                spatial_rotary_emb=self.enc_spatial_rotary_emb,
                temporal_rotary_emb=self.enc_temporal_rotary_emb
            ) for _ in range(encoder_depth)])
        # layer normalization
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # ----- Decoder -----
        # initial projection
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        # special token [MASK] for selected tokens that decoder should in-paint
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # RoPE rotary embeddings for encoder blocks along spatial and temporal dimensions
        self.dec_spatial_rotary_emb = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads // 2, freqs_for="pixel", max_freq=256)
        self.dec_temporal_rotary_emb = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)
        # decoder spatio-temporal attention blocks
        self.decoder_blocks = nn.ModuleList([
            STBlock(
                decoder_embed_dim, decoder_num_heads, qkv_bias=True,
                proj_drop=proj_dropout, attn_drop=attn_dropout, 
                spatial_rotary_emb=self.dec_spatial_rotary_emb, 
                temporal_rotary_emb=self.dec_temporal_rotary_emb
            ) for _ in range(decoder_depth)])
        # layer normalization
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # ----- pose prediction -----
        # TODO: add pose prediction network, throw into training

        # ----- special token initialization -----
        self.initialize_weights()
        
        # ----- initialize diff loss -----
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

        # ----- intialize the vae -----
        self.instantiate_vae(vae_config)
        assert isinstance(self.vae, AutoencoderKL)
        assert self.vae.latent_dim == token_embed_dim
        assert self.vae.seq_h == vae_seq_h && self.vae.seq_w == vae_seq_w

        self.seq_h, self.seq_w = self.vae.seq_h // patch_size, self.vae.seq_w // patch_size
        self.frame_seq_len = self.seq_h, self.seq_w
        # we assume here the diffusion model operates one frame at a time:
        self.diffusion_pos_emb_learned = nn.Parameter(torch.zeros(1, self.frame_seq_len, decoder_embed_dim))
        self.num_frames = num_frames
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.pred_token, std=.02)
        torch.nn.init.normal_(self.prev_token, std=.02)
        torch.nn.init.normal_(self.ctx_token, std=.02)
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
        # x = x.reshape(bsz, h_ * w_, c * p ** 2)
        x = x.reshape(bsz, h_, w_, c * p ** 2)
        return x  # [n, h, w, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p) # TODO: this should probably be changed for optim
        return x  # [n, c, h, w]
    
    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.frame_seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders, random_offset=False):
        bsz, _, embed_dim = x.shape
        mask_rate = self.mask_ratio_gen.rvs(1)[0]
        num_masked_tokens = int(np.ceil((self.frame_seq_len) * mask_rate))
        mask = torch.zeros(bsz, self.frame_seq_len, device=x.device)
        # TODO: consider moving this offset to any frame?
        #       this is 0 currently because pred frame is at start of seq
        if random_offset:
            offsets = torch.randint(high=self.num_frames, size=bsz, device=self.device)
        else:
            offsets = torch.zeros(bsz, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, self.frame_seq_len, device=x.device))
        return mask, offsets # b hw, b

    def add_token_buffers(self, x):
        ... #TODO

    def forward_encoder(self, x, actions, poses, mask, s_attn_mask=None, t_attn_mask=None):
        # x : expected to be b t h w d
        # TODO: double check this actually does across last dim
        x = self.z_proj(x)
        bsz, _, embed_dim = x.shape

        # TODO: add embs based on pos, actions, poses, want:
        #       x_i + E_i, Ei = E_ai + E_pi
        x = ...
        # TODO: double check this actually does across last dim
        x = self.z_proj_ln(x)

        for block in self.encoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)
        x = self.encoder_norm(x)

        return x
    
    def forward_decoder(self, x, actions, poses, mask, offsets, s_attn_mask=None, t_attn_mask=None):
        # x : expected to be b t h w d
        # mask: b hw
        # offsets: b
        b, t, h, w, d = x.shape

        s_mask = mask.view(b, 1, h, w)
        t_mask = (torch.arange(t, device=self.device).unsqueeze(0) == offsets.unsqueeze(1)).view(b, t, 1, 1)
        full_mask = s_mask & t_mask
        x = torch.where(full_mask.unsqueeze(-1), self.mask_token, x)

        # TODO: add embs based on pos (RoPE), actions, poses, want:
        #       x_i + E_i, Ei = E_ai + E_pi
        # THESE SHOULD BE DIFF FROM THE ENCODER ONES
        x = ...
        for block in self.decoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)
        x = self.decoder_norm(x)
    
    def forward_diffusion(self, z, tgt, mask):
        tgt = rearrange(tgt, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)
        z = rearrange(z, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1) # ad
        mask = rearrange(mask, "b s -> (b s)").repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=tgt, mask=mask)
        return loss

    def forward(self, frames, actions, poses, padding_mask=None):
        # TODO: fill this out more detail

        b= frames.shape[0]

        # 1) compress frames w/ vae
        x = rearrange(frames, "b t c h w -> (b t) c h w")
        x = self.vae.encode(frames).sample() # (b t) (h w) d
        x = self.patchify(x) # (bt) h w d
        x = rearrange(x, "(b t) h w d -> b t h w d", b=b)
        x_gt = x.clone().detach()

        # 2) gen mask
        orders = self.sample_orders(b)
        mask, offsets = self.random_masking(x, orders) # b hw, b

        # 3) construct attn_masks
        b, t, h, w, d = x.shape
        batch_idx = torch.arange(b, device=self.device)
        # --- spatial attn mask ---
        valid_hw = torch.ones(b, t, h*w)
        if padding_mask is not None:
            valid_hw &= padding_mask.unsqueeze(-1)
        s_attn_mask_dec = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        
        valid_hw[batch_idx, offsets] = mask
        s_attn_mask_enc = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_enc = rearrange(s_attn_mask_enc, "b t hw hw -> (b t) hw hw")

        # --- temporal attn mask ---
        valid_t = torch.ones(b, h*w, t, dtype=torch.bool, device=self.device)
        if padding_mask is not None:
            valid_t &= padding_mask.unsqueeze(1)
        t_attn_mask_dec = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)

        valid_t[batch_idx, mask, offsets] = False
        
        t_attn_mask_enc = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_enc = rearrange(t_attn_mask_enc, "b hw t t -> (b hw) t t")

        # 4) run encoder
        x = self.forward_encoder(x, actions, poses, s_attn_mask=s_attn_mask_enc, t_attn_mask=t_attn_mask_enc)

        # 5) run decoder
        z = self.forward_decoder(x, actions, poses, mask, offsets, s_attn_mask=s_attn_mask_dec, t_attn_mask=t_attn_mask_dec)
        # z : b t h w d

        # 6) split into tgt frame + diffuse
        z_t = x[batch_idx, offsets] # b h w d
        xt_gt = x_gt[batch_idx, offsets] # b h w d

        loss = self.forward_diffusion(z_t, xt_gt, mask)        

        return loss

    def training_step(self, batch, batch_idx):
        # TODO: parse batch, whether dict or tuple (MOVE TO DEVICE)
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # --- parse batch ---
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, C, H, W]
        batch_nframes = batch["num_frames"].to(self.device) # shape [B,]
        actions = batch["actions"].to(self.device) # shape ...
        poses = batch["poses"].to(self.device) # shape [B, T, 5]

        # --- construct attn_mask ---
        B, L = len(frames), self.num_frames * self.frame_seq_len
        # assert not torch.any(batch_nframes > self.num_frames)
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        # --- forward + backprop ---
        loss = self(frames, actions, poses, padding_mask=padding_mask)
        self.logger.log_metrics()
        self.log("train_loss", loss)
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
