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
from world_mar.modules.embeddings.timestep_embedding import TimestepEmbedder
from world_mar.modules.utils import instantiate_from_config
from world_mar.oasis_utils.vae import AutoencoderKL, format_image
from world_mar.models.diffloss import DiffLoss
import pytorch_lightning as pl
from time import time

seed = 42
torch.manual_seed(seed)

def mask_by_order(mask_len, order, bsz, seq_len):
    """
    Returns a boolean mask where the *first* `mask_len` indices of `order`
    are marked True. All others are False.
    """
    masking = torch.zeros(bsz, seq_len)
    masking.scatter_(1, order[:, :mask_len.item()], True)  # in-place set
    return masking

class WorldMAR(pl.LightningModule):
    """
    Assumptions Praccho's making:
        vae is an AutoencoderKL, future me can change this to any encoder decoder, but we like LDM's
        so we should be using it
    
    Req'd args... you need to give me a:
        - vae: thing that has an encode and decode

    """
    def __init__(
        self, 
        vae_config=None, # should be an AutoencoderKL 
        img_height=360, img_width=640, num_frames=5,
        patch_size=2, token_embed_dim=16,
        vae_seq_h=18, vae_seq_w=32,
        st_embed_dim=256,
        encoder_depth=8, encoder_num_heads=8,
        decoder_depth=8, decoder_num_heads=8,
        diffloss_w=256, diffloss_d=3, num_sampling_steps='100', diffusion_batch_mul=4,
        mask_random_frame=False,
        mask_ratio_min=0.7,
        proj_dropout=0.1,
        attn_dropout=0.1,
        gradient_clip_val=1.0,
        warmup_steps=10000, # TODO: change this depending on dataset size
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        plucker_dim = 6
        action_dim = 25
        embedding_hidden_dim = 128
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.seq_h, self.seq_w = vae_seq_h // patch_size, vae_seq_w // patch_size
        self.frame_seq_len = self.seq_h * self.seq_w

        # ----- masking statistics -----
        # ref: masking ratio used by MAR for image gen
        self.mask_random_frame = mask_random_frame
        self.mask_ratio_gen = stats.truncnorm((mask_ratio_min -1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # ----- global embeddings -----
        self.pose_embedder = nn.Sequential(
            nn.Linear(plucker_dim, embedding_hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_hidden_dim, st_embed_dim)
        )
        self.timestamp_embedder = TimestepEmbedder(hidden_size=st_embed_dim)
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, embedding_hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_hidden_dim, st_embed_dim)
        )

        # ----- local RoPE embeddings -----
        self.enc_spatial_rotary_embedder = RotaryEmbedding(dim=st_embed_dim // encoder_num_heads // 2, freqs_for="pixel", max_freq=256)
        self.enc_temporal_rotary_embedder = RotaryEmbedding(dim=st_embed_dim // encoder_num_heads)
        self.dec_spatial_rotary_embedder = RotaryEmbedding(dim=st_embed_dim // decoder_num_heads // 2, freqs_for="pixel", max_freq=256)
        self.dec_temporal_rotary_embedder = RotaryEmbedding(dim=st_embed_dim // decoder_num_heads)

        # ----- encoder -----
        # initial projection
        self.patch_size = patch_size
        self.token_embed_dim = token_embed_dim * patch_size**2
        self.z_proj = nn.Linear(self.token_embed_dim, st_embed_dim, bias=True) # projs VAE latents to transformer dim
        self.z_proj_ln = nn.LayerNorm(st_embed_dim, eps=1e-6)
        # special tokens [PRED] and [CTX]
        self.pred_token = nn.Parameter(torch.zeros(1, st_embed_dim))
        self.ctx_token = nn.Parameter(torch.zeros(1, st_embed_dim))
        # encoder spatio-temporal attention blocks
        self.encoder_blocks = nn.ModuleList([
            STBlock(
                st_embed_dim, encoder_num_heads, qkv_bias=True,
                proj_drop=proj_dropout, attn_drop=attn_dropout,
                spatial_rotary_emb=self.enc_spatial_rotary_embedder,
                temporal_rotary_emb=self.enc_temporal_rotary_embedder
            ) for _ in range(encoder_depth)])
        # layer normalization
        self.encoder_norm = nn.LayerNorm(st_embed_dim)

        # ----- decoder -----
        # special token [MASK] for selected tokens that decoder should in-paint
        self.mask_token = nn.Parameter(torch.zeros(1, 1, st_embed_dim))
        # decoder spatio-temporal attention blocks
        self.decoder_blocks = nn.ModuleList([
            STBlock(
                st_embed_dim, decoder_num_heads, qkv_bias=True,
                proj_drop=proj_dropout, attn_drop=attn_dropout, 
                spatial_rotary_emb=self.dec_spatial_rotary_embedder,
                temporal_rotary_emb=self.dec_temporal_rotary_embedder
            ) for _ in range(decoder_depth)])
        # layer normalization
        self.decoder_norm = nn.LayerNorm(st_embed_dim)

        # ----- pose prediction -----
        # TODO: add pose prediction network, throw into training

        # ----- special token initialization -----
        self.initialize_weights()
        
        # ----- initialize diff loss -----
        # TODO: make cutomizable as MLP (per patch?) vs DiT (per frame).
        #       for now, assuming more lightweight MLP
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=st_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        # ----- intialize the vae -----
        if vae_config:
            self.instantiate_vae(vae_config)
            assert isinstance(self.vae, AutoencoderKL)
            self.vae_embed_dim = self.vae.latent_dim
            assert self.vae.latent_dim == token_embed_dim
            assert self.vae.seq_h == vae_seq_h and self.vae.seq_w == vae_seq_w

        self.vae_seq_h = 18
        self.vae_seq_w = 32

        self.seq_h, self.seq_w = self.vae_seq_h // patch_size, self.vae_seq_w // patch_size
        self.frame_seq_len = self.seq_h * self.seq_w
        # we assume here the diffusion model operates one frame at a time:
        self.diffusion_pos_emb_learned = nn.Parameter(torch.zeros(1, self.seq_h, self.seq_w, st_embed_dim))
        torch.nn.init.kaiming_normal_(self.diffusion_pos_emb_learned)
        self.num_frames = num_frames
    
    @staticmethod
    def load_filtered_checkpoint(filepath, map_location=None, **kwargs):
        checkpoint = torch.load(filepath, map_location=map_location)
        # Remove vae weights
        state_dict = checkpoint['state_dict']
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith('vae')
        }
        checkpoint['state_dict'] = filtered_state_dict

        # Now load the module
        model = WorldMAR(**kwargs)
        model.load_state_dict(filtered_state_dict, strict=False)
        return model

    def initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.pred_token)
        torch.nn.init.kaiming_normal_(self.ctx_token)
        torch.nn.init.kaiming_normal_(self.mask_token)
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
        self.vae.to(self.device)
        self.vae.train = lambda self, mode=True: self
        for param in self.vae.parameters():
            param.requires_grad=False
    
    def patchify(self, x):
        bsz, s, c = x.shape
        p = self.patch_size
        h_, w_ = self.seq_h, self.seq_w
        x = rearrange(x, "b (h w) c -> b c h w", h=h_*p)

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        # x = x.reshape(bsz, h_ * w_, c * p ** 2)
        x = x.reshape(bsz, h_, w_, c * p ** 2)
        return x  # [bsz, h, w, d]

    def unpatchify(self, x):
        # b (h w) d -> b d/p**2 (h p w p)
        
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        # x = torch.einsum('nhwcpq->nchpwq', x) # (n, h, w, c, p, q)  →  (n, c, h, p, w, q)
        x = rearrange(x, "n h w c p q -> n h p w q c")
        x = x.reshape(bsz, h_ * p * w_ * p, c)
        return x  # [bsz, c, (h w)]
    
    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.frame_seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        # orders = torch.Tensor(np.array(orders)).cuda().long()
        orders = torch.Tensor(np.array(orders)).to(self.device).long()
        return orders

    def random_masking(self, x, orders, random_offset=False):
        bsz= x.shape[0]
        mask_rate = self.mask_ratio_gen.rvs(1)[0]
        num_masked_tokens = int(np.ceil(self.frame_seq_len * mask_rate))
        mask = torch.zeros(bsz, self.frame_seq_len, dtype=bool, device=x.device)
        # TODO: consider moving this offset to any frame?
        #       this is 0 currently because pred frame is at start of seq
        if random_offset:
            # TODO: THIS IS WRONG, HIGH SHOULD BE BATCH ITEM NUM FRAME DEPENDENT
            offsets = torch.randint(high=self.num_frames, size=bsz, dtype=torch.int64, device=self.device)
        else:
            offsets = torch.zeros(bsz, dtype=torch.int64, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, self.frame_seq_len, dtype=bool, device=x.device))
        return mask, offsets # b hw, b

    def add_pose_and_timestamp_embeddings(self, x, poses, timestamps, is_decoder=False):
        B, T, H, W, D = x.shape

        if is_decoder:
            H -= 1  # encoder would have concatenated buffer onto x's H dim
            
        assert timestamps.shape == (B, T)

        # construct pose embeddings matching latent shape
        poses = poses[:, :, ::36, ::40, :]           # (B, T, H, W, 6)
        pose_embeddings = self.pose_embedder(poses)  # (B, T, H, W, D)

        # construct timestamp embeddings matching latent shape
        timestamps = rearrange(timestamps, "b t -> (b t)")                               # (BT)
        timestamp_embeddings = self.timestamp_embedder(timestamps)                       # (BT, D)
        timestamp_embeddings = rearrange(timestamp_embeddings, "(b t) d -> b t d", b=B)  # (B, T, D)
        timestamp_embeddings = timestamp_embeddings.unsqueeze(2).unsqueeze(3)            # (B, T, 1, 1, D)
        timestamp_embeddings = timestamp_embeddings.expand((-1, -1, H, W, -1))           # (B, T, H, W, D)

        # return the latent with the embeddings added
        x[:, :, :H, :, :] += pose_embeddings + timestamp_embeddings  # (B, T, H or H+1, W, D)
        return x

    def add_pred_action_and_ctx_embeddings(self, x, actions, is_decoder=False):
        B, T, H, W, D = x.shape

        if is_decoder:
            H -= 1  # encoder would have concatenated buffer onto x's H dim

        assert actions.shape == (B, 25)

        # calculate action tokens
        action_embeddings = self.action_embedder(actions)  # (B, D)

        # tokens: expected to be d-dimensional vectors
        pred_token_buffer = self.pred_token.view(1, 1, 1, 1, D).repeat(B, 1, 1, W, 1)      # (B, 1, 1, W, D)
        action_token_buffer = action_embeddings.view(B, 1, 1, 1, D).repeat(1, 1, 1, W, 1)  # (B, 1, 1, W, D)
        ctx_token_buffer = self.ctx_token.view(1, 1, 1, 1, D).repeat(B, T-2, 1, W, 1)      # (B, T-2, 1, W, D)

        if not is_decoder:
            # concat [PRED] token buffer to x[:, 0] (first elements along temporal dim) 
            # concat [ACTION] token buffer to x[:, 1] (second elements along temporal dim)
            # concat [CTX] token buffer to x[:, 2:] (third, fourth, fifth, ... elements along temporal dim)
            # these concatenations are happening along H, the height dimension (could've been W equivalently)
            x = torch.cat([
                torch.cat([x[:, 0:1], pred_token_buffer], dim=-3),
                torch.cat([x[:, 1:2], action_token_buffer], dim=-3),
                torch.cat([x[:, 2:], ctx_token_buffer], dim=-3)
            ], dim=-4)  # dim=-3 is H and dim=-4 is T
        else:
            # add [PRED] tokens to x[:, 0, H:, :, :] (extra buffer for first elements along temporal dim) 
            # add [ACTION] tokens to x[:, 1, H:, :, :] (extra buffer for second elements along temporal dim)
            # add [CTX] tokens to x[:, 2:, H:, :, :] (extra buffer for third, fourth, fifth, ... elements along temporal dim)
            tokens = torch.cat([pred_token_buffer, action_token_buffer, ctx_token_buffer], dim=-4)  # (B, T, 1, W, D)
            x[:, :, H:, :, :] += tokens  # (B, T, H+1, W, D)

        return x  # (B, T, H+1, W, D)

    def forward_encoder(self, x, actions, poses, timestamps, s_attn_mask=None, t_attn_mask=None):
        # x:       (B, T, H, W, token_embed_dim)
        # actions: (B, 25)
        # poses:   (B, T, 40H, 40W, 6)
        # TODO: double check this actually does across last dim
        x = self.z_proj(x)  # (B, T, H, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_and_timestamp_embeddings(x, poses, timestamps)  # (B, T, H, W, D)

        # add token buffers for prediction, action (on prev frame), and context (on memory frames)
        x = self.add_pred_action_and_ctx_embeddings(x, actions)  # (B, T, H+1, W, D)

        # pass through each encoder spatio-temporal attention block
        # TODO: double check this actually does across last dim
        x = self.z_proj_ln(x)

        for block in self.encoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)

        x = self.encoder_norm(x) 

        return x  # (B, T, H+1, W, D)

    def forward_decoder(self, x, actions, poses, timestamps, mask, offsets, s_attn_mask=None, t_attn_mask=None):
        # x:       (B, T, H+1, W, D) 
        # mask:    (B, HW)
        # offsets: (B)
        B, T, H, W, D = x.shape
        H -= 1  # encoder would have concatenated token buffer onto x's H dim

        # convert mask and offsets into separate space and time masks
        s_mask = mask.view(B, 1, H+1, W)
        t_mask = (torch.arange(T, device=self.device).unsqueeze(0) == offsets.unsqueeze(1)).view(B, T, 1, 1)
        full_mask = s_mask & t_mask
        x = torch.where(full_mask.unsqueeze(-1), self.mask_token, x)  # (B, T, H+1, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_and_timestamp_embeddings(x, poses, timestamps, is_decoder=True)  # (B, T, H+1, W, D)

        # re-add [PRED], [ACTION], and [CTX] tokens to the token buffers
        x = self.add_pred_action_and_ctx_embeddings(x, actions, is_decoder=True)  # (B, T, H+1, W, D)

        # pass through each decoder spatio-temporal attention block
        for block in self.decoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)

        x = self.decoder_norm(x)

        # remove the extra token buffer that was concatenated by the encoder onto x's H dim
        x = x[:, :, :H, :, :]

        return x  # (B, T, H, W, D)
    
    def forward_diffusion(self, z, tgt, mask):
        tgt = rearrange(tgt, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)
        z = z + self.diffusion_pos_emb_learned
        z = rearrange(z, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1) # ad
        mask = rearrange(mask, "b s -> (b s)").repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=tgt, mask=mask)
        return loss

    def construct_attn_masks(self, x, mask, offsets, padding_mask=None):
        b, t, h, w, d = x.shape
        batch_idx = torch.arange(b, device=self.device)

        # --- spatial attn mask ---
        valid_hw = torch.ones(b, t, (h+1)*w, dtype=torch.bool, device=self.device)
        if padding_mask is not None:
            valid_hw &= padding_mask.unsqueeze(-1)
        s_attn_mask_dec = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_dec = rearrange(s_attn_mask_dec, "b t hw1 hw2 -> (b t) 1 hw1 hw2")
        valid_hw[batch_idx, offsets] = ~mask
        
        s_attn_mask_enc = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_enc = rearrange(s_attn_mask_enc, "b t hw1 hw2 -> (b t) 1 hw1 hw2")

        # --- temporal attn mask ---
        valid_t = torch.ones(b, (h+1)*w, t, dtype=torch.bool, device=self.device)
        if padding_mask is not None:
            valid_t &= padding_mask.unsqueeze(1)
        t_attn_mask_dec = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_dec = rearrange(t_attn_mask_dec, "b hw t1 t2 -> (b hw) 1 t1 t2")

        _, l, _ = valid_t.shape

        batch_idx = batch_idx.unsqueeze(1).expand(-1, l)
        len_idx   = torch.arange(l, device=valid_t.device).unsqueeze(0).expand(b, -1)
        depth_idx = offsets.unsqueeze(1).expand(-1, l)
        valid_t[batch_idx[mask], len_idx[mask], depth_idx[mask]] = False
        
        t_attn_mask_enc = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_enc = rearrange(t_attn_mask_enc, "b hw t1 t2 -> (b hw) 1 t1 t2")

        return s_attn_mask_enc, t_attn_mask_enc, s_attn_mask_dec, t_attn_mask_dec

    def _compute_z_and_mask(self, x, actions, poses, timestamps, padding_mask=None):
        b = x.shape[0]

        # 1) patchify latents
        x = rearrange(x, "b t s c -> (b t) s c")
        x = self.patchify(x) # (bt) h w d (different h and w bc of patchifying)
        x = rearrange(x, "(b t) h w d -> b t h w d", b=b)
        x_gt = x.clone().detach()

        # 2) gen mask
        b, t, h, w, d = x.shape
        orders = self.sample_orders(b)
        mask, offsets = self.random_masking(x, orders, random_offset=self.mask_random_frame) # b hw, b
        pad_mask = rearrange(mask, "b (h w) -> b h w", h=h)
        pad_mask = torch.cat([pad_mask, torch.zeros(b,1,w, dtype=torch.bool, device=self.device)], dim=-2)
        pad_mask = rearrange(pad_mask, "b h w -> b (h w)")

        # 3) construct attn_masks
        (s_attn_mask_enc, t_attn_mask_enc, 
         s_attn_mask_dec, t_attn_mask_dec) = self.construct_attn_masks(x, pad_mask, offsets, padding_mask=padding_mask)

        # 4) run encoder
        x = self.forward_encoder(x, actions, poses, timestamps, s_attn_mask=s_attn_mask_enc, t_attn_mask=t_attn_mask_enc)

        # 5) run decoder
        z = self.forward_decoder(x, actions, poses, timestamps, pad_mask, offsets, s_attn_mask=s_attn_mask_dec, t_attn_mask=t_attn_mask_dec)
        # z : b t h w d

        return z, mask, offsets, x_gt

    def forward(self, x, actions, poses, timestamps, padding_mask=None):
        b = x.shape[0]
        
        z, mask, offsets, x_gt = self._compute_z_and_mask(x, actions, poses, timestamps, padding_mask)

        # split into tgt frame + diffuse
        batch_idx = torch.arange(b, device=self.device)
        z_t = z[batch_idx, offsets] # b h w d
        xt_gt = x_gt[batch_idx, offsets] # b h w d

        loss = self.forward_diffusion(z_t, xt_gt, mask)        

        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # print("START OF TRAINING STEP")

        # --- parse batch ---
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, H, W, C]
        batch_nframes = batch["num_non_padding_frames"].to(self.device) # shape [B,]
        actions = batch["action"].to(self.device) # shape [B, 25]
        poses = batch["plucker"].to(self.device) # shape [B, T, H, W, 6]
        timestamps = batch["timestamps"].to(self.device) # shape [B, T]

        start = time()

        # --- construct padding_mask ---
        B, L = len(frames), self.num_frames * self.frame_seq_len
        # assert not torch.any(batch_nframes > self.num_frames)
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        # --- forward + backprop ---
        loss = self(frames, actions, poses, timestamps, padding_mask=padding_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # --- clip gradients, backwards, step
        # def max_gradients(params):
        #     return max([p.grad.abs().max().item() for p in params if p.grad is not None], default=0.0)
        
        self.manual_backward(loss)
        # print(f"(Before clipping) Maximum of gradients: {max_gradients(self.parameters())}")
        grad_update_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val, norm_type=2)
        # print(f"(After clipping) Maximum of gradients: {max_gradients(self.parameters())}")
        # print(f"Total norm of gradient update vector: {grad_update_norm}")

        opt.step()
        lr_sched.step()

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate)
        lr_sched = LinearLR(optim, start_factor=1.0/5, end_factor=1.0, total_iters=self.warmup_steps)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "interval": "step",
                "frequency": 1,
            }
        }

    def sample(self, x, actions, poses, timestamps, batch_nframes):
        
        # --- parse batch ---
        # assume the layout is [PREV_FRAME, CTX_FRAMES ...]
        # frames = format_image(batch["frames"].to(self.device)) # shape [B, T, H, W, C]
        # batch_nframes = batch["num_non_padding_frames"].to(self.device) # shape [B,]
        # actions = batch["action"].to(self.device) # shape [B, 25]
        # poses = batch["plucker"].to(self.device) # shape [B, T, H, W, 6]
        # timestamps = batch["timestamps"].to(self.device) # shape [B, T]

        # --- construct padding_mask ---
        B, L = len(x), self.num_frames * self.frame_seq_len
        # assert not torch.any(batch_nframes > self.num_frames)
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        z, mask, offsets, x_gt = self._compute_z_and_mask(x, actions, poses, timestamps, padding_mask=padding_mask)

        # grab tokens for [MASK] frame
        batch_idx = torch.arange(B, device=self.device)
        z_mask = z[batch_idx, offsets] # b h w d
        # xt_gt = x_gt[batch_idx, offsets] # b h w d

        # diffuse
        z_mask = z_mask + self.diffusion_pos_emb_learned
        z_mask = rearrange(z_mask, "b h w d -> (b h w) d")
        
        # bc we're predicting on all masked at once, this is pretty simple
        start = time()
        patch_preds = self.diffloss.sample_ddim(z_mask, cfg=1.0) # (b h w) d
        end = time()
        patch_preds = rearrange(patch_preds, "(b h w) d -> b (h w) d", b=B, h=self.seq_h, w=self.seq_w)
        # print("out:", patch_preds.shape, end - start)
        x_pred = self.unpatchify(patch_preds)

        return x_pred

    # def sample_tokens(self, bsz, num_iter=64, labels=None, temperature=1.0, progress=False):

    #     # init and sample generation orders
    #     mask = torch.ones(bsz, self.seq_len).cuda()
    #     tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
    #     orders = self.sample_orders(bsz)

    #     indices = list(range(num_iter))
    #     if progress:
    #         indices = tqdm(indices)
    #     # generate latents
    #     for step in indices:
    #         cur_tokens = tokens.clone()

    #         # class embedding and CFG
    #         if labels is not None:
    #             class_embedding = self.class_emb(labels)
    #         else:
    #             class_embedding = self.fake_latent.repeat(bsz, 1)
    #         if not cfg == 1.0:
    #             tokens = torch.cat([tokens, tokens], dim=0)
    #             class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
    #             mask = torch.cat([mask, mask], dim=0)

    #         # mae encoder
    #         x = self.forward_mae_encoder(tokens, mask, class_embedding)

    #         # mae decoder
    #         z = self.forward_mae_decoder(x, mask)

    #         # mask ratio for the next round, following MaskGIT and MAGE.
    #         mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
    #         mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

    #         # masks out at least one for the next iteration
    #         mask_len = torch.maximum(torch.Tensor([1]).cuda(),
    #                                  torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

    #         # get masking for next iteration and locations to be predicted in this iteration
    #         mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
    #         if step >= num_iter - 1:
    #             mask_to_pred = mask[:bsz].bool()
    #         else:
    #             mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
    #         mask = mask_next
    #         if not cfg == 1.0:
    #             mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

    #         # sample token latents for this step
    #         z = z[mask_to_pred.nonzero(as_tuple=True)]
    #         # cfg schedule follow Muse
    #         if cfg_schedule == "linear":
    #             cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
    #         elif cfg_schedule == "constant":
    #             cfg_iter = cfg
    #         else:
    #             raise NotImplementedError
    #         sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
    #         if not cfg == 1.0:
    #             sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
    #             mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

    #         cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
    #         tokens = cur_tokens.clone()

    #     # unpatchify
    #     tokens = self.unpatchify(tokens)
    #     return tokens
