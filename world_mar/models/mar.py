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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
        mask_random_frame=False, prev_masking_rate=0.5,
        mask_ratio_min=0.7,
        proj_dropout=0.1,
        attn_dropout=0.1,
        gradient_clip_val=1.0,
        warmup_steps=13500,  # TODO: change this depending on dataset size
        **kwargs
    ):
        super().__init__()
        self.automatic_optimization = False
        plucker_dim = 6
        action_dim = 25
        embedding_hidden_dim = 256
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.seq_h, self.seq_w = vae_seq_h // patch_size, vae_seq_w // patch_size
        self.frame_seq_len = self.seq_h * self.seq_w

        # ----- masking statistics -----
        # ref: masking ratio used by MAR for image gen
        self.mask_random_frame = mask_random_frame
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.prev_masking_rate = prev_masking_rate

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
        # self.z_proj_ln = nn.LayerNorm(st_embed_dim)

        # special token [PRED] for final frame that must be generated
        self.pred_token = nn.Parameter(torch.zeros(1, st_embed_dim))

        # encoder spatio-temporal attention blocks
        self.encoder_blocks = nn.ModuleList([
            STBlock(
                st_embed_dim, encoder_num_heads, qkv_bias=True,
                proj_drop=proj_dropout, attn_drop=attn_dropout,
                spatial_rotary_emb=self.enc_spatial_rotary_embedder,
                temporal_rotary_emb=self.enc_temporal_rotary_embedder
            ) for _ in range(encoder_depth)])

        # final normalization
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
        
        # final normalization
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
        self.scale_factor = 0.09

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
        # b (h w) d -> b (h p w p) d/p**2
        
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        # x = torch.einsum('nhwcpq->nchpwq', x) # (n, h, w, c, p, q)  →  (n, c, h, p, w, q)
        x = rearrange(x, "n h w c p q -> n h p w q c")
        x = x.reshape(bsz, h_ * p * w_ * p, c)
        return x  # [bsz, (h w), c]
    
    def sample_orders(self, batch_size):
        # contains B randomized vectors of indices going from 0, 1, ..., HW-1
        orders = []

        for _ in range(batch_size):
            order = np.array(list(range(self.frame_seq_len)))           # (HW)
            np.random.shuffle(order)                                    # (HW)
            orders.append(order)

        orders = torch.Tensor(np.array(orders)).to(self.device).long()  # (B, HW)
        return orders

    def random_masking(self, x, orders, random_offset=False, masking_rate=None):
        B = x.shape[0]

        mask_rate = self.mask_ratio_generator.rvs(1)[0] if not masking_rate else masking_rate
        num_masked_tokens = int(np.ceil(self.frame_seq_len * mask_rate))
        mask = torch.zeros((B, self.frame_seq_len), dtype=bool, device=x.device)  # (B, HW)

        if random_offset:
            # All frames have random masking as regularlization; pick a random frame to actually 
            # predict over (masks on that particular frame become tokens that are diffused over)
            offsets = torch.randint(high=self.num_frames, size=B, dtype=torch.int64, device=self.device)  # (B)
        else:
            # Predict the last frame (at index 0)
            offsets = torch.zeros(B, dtype=torch.int64, device=x.device)                                  # (B)

        pred_mask = torch.scatter(
            mask,
            dim=-1, 
            index=orders[:, :num_masked_tokens], 
            src=torch.ones(B, self.frame_seq_len, dtype=bool, device=x.device)
        )  # (B, HW)
        
        if self.training and not random_offset:
            prev_mask = torch.scatter(
                mask, 
                dim=-1, 
                index=orders[:, :int(num_masked_tokens*self.prev_masking_rate)],
                src=torch.ones(B, self.frame_seq_len, dtype=bool, device=x.device)
            )
        else:
            prev_mask = None

        return pred_mask, prev_mask, offsets  # (B, HW), (B, HW), (B)

    def add_pose_and_timestamp_embeddings(self, x, poses, timestamps, is_decoder=False):
        B, T, H, W, D = x.shape

        if is_decoder:
            H -= 1  # encoder would have concatenated buffer onto x's H dim
            
        assert timestamps.shape == (B, T)
        assert poses.shape[:2] == (B, T) and poses.shape[-1] == 6

        # construct pose embeddings matching latent shape
        poses = poses[:, :, ::36, ::40, :]           # (B, T, H, W, 6)
        pose_embeddings = self.pose_embedder(poses)  # (B, T, H, W, D)
        assert pose_embeddings.shape == (B, T, H, W, D)

        # construct timestamp embeddings matching latent shape
        timestamps = rearrange(timestamps, "b t -> (b t)")                               # (BT)
        timestamp_embeddings = self.timestamp_embedder(timestamps)                       # (BT, D)
        timestamp_embeddings = rearrange(timestamp_embeddings, "(b t) d -> b t d", b=B)  # (B, T, D)
        timestamp_embeddings = timestamp_embeddings.unsqueeze(2).unsqueeze(3)            # (B, T, 1, 1, D)
        timestamp_embeddings = timestamp_embeddings.expand((-1, -1, H, W, -1))           # (B, T, H, W, D)
        assert timestamp_embeddings.shape == (B, T, H, W, D)

        # return the latent with the embeddings added
        x[:, :, :H, :, :] = x[:, :, :H, :, :] + pose_embeddings + timestamp_embeddings   # (B, T, H or H+1, W, D)
        return x

    def add_pred_and_action_embeddings(self, x, actions, is_decoder=False):
        B, T, H, W, D = x.shape

        if is_decoder:
            H -= 1  # encoder would have concatenated buffer onto x's H dim

        assert actions.shape == (B, T-1, 25)

        # calculate action tokens
        action_embeddings = self.action_embedder(actions)                                    # (B, T-1, D)

        # construct [PRED] and [ACTION] token buffers (expected to be a slice varying along time-dimension)
        pred_token_buffer = self.pred_token.view(1, 1, 1, 1, D).repeat(B, 1, 1, W, 1)        # (B, 1, 1, W, D)
        action_token_buffer = action_embeddings.view(B, T-1, 1, 1, D).repeat(1, 1, 1, W, 1)  # (B, T-1, 1, W, D)

        if not is_decoder:
            # concat [PRED] token buffer to x[:, 0] (first elements along temporal dim) 
            # concat [ACTION] token buffer to x[:, 1:] (second, third, fourth, ... elements along temporal dim)
            # these concatenations are happening along H, the height dimension (could've been W equivalently)
            x = torch.cat([
                torch.cat([x[:, 0:1], pred_token_buffer], dim=-3),
                torch.cat([x[:, 1:], action_token_buffer], dim=-3)  # (B, T, H+1, W, D)
            ], dim=-4)  # dim=-3 is H and dim=-4 is T
        else:
            # add [PRED] tokens to x[:, 0, H:, :, :] (extra buffer for first elements along temporal dim) 
            # add [ACTION] tokens to x[:, 1:, H:, :, :] (extra buffer for second, third, fourth, ... elements along temporal dim)
            tokens = torch.cat([pred_token_buffer, action_token_buffer], dim=-4)  # (B, T, 1, W, D)
            x[:, :, H:, :, :] = x[:, :, H:, :, :] + tokens                        # (B, T, H+1, W, D)

        assert x.shape == (B, T, H+1, W, D)
        return x  # (B, T, H+1, W, D)

    def forward_encoder(self, x, num_ctx_frames, actions, poses, timestamps, s_attn_mask=None, t_attn_mask=None):
        # x:       (B, T, H, W, token_embed_dim)
        # actions: (B, 25)
        # poses:   (B, T, 40H, 40W, 6)

        # project and layer normalize
        x = self.z_proj(x)                                                  # (B, T, H, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_and_timestamp_embeddings(x, poses, timestamps)    # (B, T, H, W, D)

        # add token buffers for prediction, action (on prev frame), and context (on memory frames)
        x = self.add_pred_and_action_embeddings(x, actions)                 # (B, T, H+1, W, D)
        x = self.z_proj_ln(x)                                               # (B, T, H+1, W, D)

        # pass through each encoder spatio-temporal attention block
        for block in self.encoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)  # (B, T, H+1, W, D)
        
        # final layer norm before passing to the decoder
        x = self.encoder_norm(x)                                            # (B, T, H+1, W, D)

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
        x = torch.where(full_mask.unsqueeze(-1), self.mask_token, x)                       # (B, T, H+1, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_and_timestamp_embeddings(x, poses, timestamps, is_decoder=True)  # (B, T, H+1, W, D)

        # re-add [PRED], [ACTION], and [CTX] tokens to the token buffers
        x = self.add_pred_and_action_embeddings(x, actions, is_decoder=True)               # (B, T, H+1, W, D)

        # pass through each decoder spatio-temporal attention block
        for block in self.decoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)                 # (B, T, H+1, W, D)

        # final layer norm before passing to the diffusion backbone
        x = self.decoder_norm(x)                                                           # (B, T, H+1, W, D)

        # remove the extra token buffer that was concatenated by the encoder onto x's H dim
        x = x[:, :, :H, :, :]                                                              # (B, T, H, W, D)

        return x  # (B, T, H, W, D)
    
    def forward_diffusion(self, z, target, mask):
        target = rearrange(target, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)  # (4BHW, D)
        z = z + self.diffusion_pos_emb_learned
        z = rearrange(z, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)            # (4BHW, D)
        mask = rearrange(mask, "b s -> (b s)").repeat(self.diffusion_batch_mul)                 # (4BHW)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def construct_attn_masks(self, x, mask, offsets, prev_mask=None, padding_mask=None):
        b, t, h, w, d = x.shape
        batch_idx = torch.arange(b, device=self.device)

        # --- spatial attn mask ---
        valid_hw = torch.ones(b, t, (h+1)*w, dtype=torch.bool, device=self.device)

        if padding_mask is not None:
            valid_hw &= padding_mask.unsqueeze(-1)

        if prev_mask is not None:
            offsets_prev = torch.ones_like(offsets, dtype=torch.int64, device=self.device)
            valid_hw[batch_idx, offsets_prev] = ~prev_mask

        s_attn_mask_dec = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_dec = rearrange(s_attn_mask_dec, "b t hw1 hw2 -> (b t) 1 hw1 hw2")
        valid_hw[batch_idx, offsets] = ~mask
        
        s_attn_mask_enc = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_enc = rearrange(s_attn_mask_enc, "b t hw1 hw2 -> (b t) 1 hw1 hw2")

        # --- temporal attn mask ---
        valid_t = torch.ones(b, (h+1)*w, t, dtype=torch.bool, device=self.device)
        _, l, _ = valid_t.shape
        batch_idx = batch_idx.unsqueeze(1).expand(-1, l)
        len_idx   = torch.arange(l, device=valid_t.device).unsqueeze(0).expand(b, -1)
        depth_idx = offsets.unsqueeze(1).expand(-1, l)

        if padding_mask is not None:
            valid_t &= padding_mask.unsqueeze(1)

        if prev_mask is not None:
            prev_depth_idx = offsets_prev.unsqueeze(1).expand(-1, l)
            valid_t[batch_idx[prev_mask], len_idx[prev_mask], prev_depth_idx[prev_mask]] = False

        t_attn_mask_dec = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_dec = rearrange(t_attn_mask_dec, "b hw t1 t2 -> (b hw) 1 t1 t2")

        valid_t[batch_idx[mask], len_idx[mask], depth_idx[mask]] = False
        
        t_attn_mask_enc = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_enc = rearrange(t_attn_mask_enc, "b hw t1 t2 -> (b hw) 1 t1 t2")

        return s_attn_mask_enc, t_attn_mask_enc, s_attn_mask_dec, t_attn_mask_dec

    def masked_encoder_decoder(self, x, actions, poses, timestamps, padding_mask=None, masking_rate=None):
        B = x.shape[0]

        # 1) patchify latents
        x = rearrange(x, "b t s c -> (b t) s c")
        x = self.patchify(x) # (bt) h w d (different h and w bc of patchifying)
        x = rearrange(x, "(b t) h w d -> b t h w d", b=B)
        x_gt = x.clone().detach()

        # 2) gen mask
        B, T, H, W, D = x.shape
        orders = self.sample_orders(B)
        pred_mask, prev_mask, offsets = self.random_masking(x, orders, random_offset=self.mask_random_frame, masking_rate=masking_rate) # b hw, b
        spatial_mask = rearrange(pred_mask, "b (h w) -> b h w", h=H)
        spatial_mask = torch.cat([spatial_mask, torch.zeros(B, 1, W, dtype=torch.bool, device=self.device)], dim=-2)
        spatial_mask = rearrange(spatial_mask, "b h w -> b (h w)")

        if prev_mask is not None:
            prev_mask = rearrange(prev_mask, "b (h w) -> b h w", h=H)
            prev_mask = torch.cat([prev_mask, torch.zeros(B, 1, W, dtype=torch.bool, device=self.device)], dim=-2)
            prev_mask = rearrange(prev_mask, "b h w -> b (h w)")

        # 3) construct attn_masks
        (s_attn_mask_enc, t_attn_mask_enc, s_attn_mask_dec, t_attn_mask_dec) = self.construct_attn_masks(
            x, spatial_mask, offsets, prev_mask=prev_mask, padding_mask=padding_mask
        )

        # 4) run encoder
        x = self.forward_encoder(x, actions, poses, timestamps, s_attn_mask=s_attn_mask_enc, t_attn_mask=t_attn_mask_enc)

        # 5) run decoder
        z = self.forward_decoder(x, actions, poses, timestamps, spatial_mask, offsets, s_attn_mask=s_attn_mask_dec, t_attn_mask=t_attn_mask_dec)
        # z : b t h w d

        return z, pred_mask, offsets, x_gt

    def forward(self, x, actions, poses, timestamps, padding_mask=None):
        b = x.shape[0]

        # scale the input tensor x to a standard normal distribution
        x = x * self.scale_factor
        
        # pass through the main masked spatio-temporal attention mechanism
        z, mask, offsets, x_gt = self.masked_encoder_decoder(x, actions, poses, timestamps, padding_mask)

        # split into target frame + diffuse
        batch_idx = torch.arange(b, device=self.device)
        z_t = z[batch_idx, offsets] # b h w d
        xt_gt = x_gt[batch_idx, offsets] # b h w d

        loss = self.forward_diffusion(z_t, xt_gt, mask)        

        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # --- parse batch ---
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, H, W, C]
        batch_nframes = batch["num_non_padding_frames"].to(self.device) # shape [B,]
        actions = batch["action"].to(self.device) # shape [B, 25]
        poses = batch["plucker"].to(self.device) # shape [B, T, H, W, 6]
        timestamps = batch["timestamps"].to(self.device) # shape [B, T]

        # --- construct padding_mask ---
        B, L = len(frames), self.num_frames * self.frame_seq_len
        # assert not torch.any(batch_nframes > self.num_frames)
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        # --- forward + backprop ---
        loss = self(frames, actions, poses, timestamps, padding_mask=padding_mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # --- clip gradients, backwards, step
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.gradient_clip_val, norm_type=2)

        opt.step()
        lr_sched.step()

    def validation_step(self, batch, batch_idx):
        # --- parse batch ---
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, H, W, C]
        batch_nframes = batch["num_non_padding_frames"].to(self.device) # shape [B,]
        actions = batch["action"].to(self.device) # shape [B, 25]
        poses = batch["plucker"].to(self.device) # shape [B, T, H, W, 6]
        timestamps = batch["timestamps"].to(self.device) # shape [B, T]

        # --- construct padding_mask ---
        B, L = len(frames), self.num_frames * self.frame_seq_len
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        # --- forward + backprop ---
        loss = self(frames, actions, poses, timestamps, padding_mask=padding_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate)
        warmup_sched = LinearLR(
            optim,
            start_factor=2e-1,   # or 0.0
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine_sched = CosineAnnealingLR(
            optim,
            T_max=self.warmup_steps * 50 - self.warmup_steps,
            eta_min=0.0
        )
        scheduler = SequentialLR(
            optim,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[self.warmup_steps]
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
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

        # scale the input tensor x to match a standard normal distribution
        x = x * self.scale_factor

        # assert not torch.any(batch_nframes > self.num_frames)
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        z, mask, offsets, x_gt = self.masked_encoder_decoder(x, actions, poses, timestamps, padding_mask=padding_mask, masking_rate=1.0)

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

        # undo the scaling operation done to the input tensor (recover the original range of values)
        x_pred = x_pred / self.scale_factor

        return x_pred
