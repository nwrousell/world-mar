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
        num_mem_frames=2, num_prev_frames=3,  # NOTE: num_prev_frames here is 1 more than in dataloader because of the target frame
        patch_size=2, vae_embed_dim=16,
        vae_seq_h=18, vae_seq_w=32,
        st_embed_dim=256,
        encoder_depth=8, encoder_num_heads=8,
        decoder_depth=8, decoder_num_heads=8,
        diffloss_w=256, diffloss_d=3, num_sampling_steps='100', diffusion_batch_mul=4,
        prev_masking_rate=0.5,
        mask_ratio_min=0.7,
        proj_dropout=0.1,
        attn_dropout=0.1,
        gradient_clip_val=1.0,
        warmup_steps=40_000,  # TODO: change this depending on dataset size
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
        self.num_mem_frames = num_mem_frames
        self.num_prev_frames = num_prev_frames

        # ----- masking statistics -----
        # ref: masking ratio used by MAR for image gen
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
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.z_proj = nn.Linear(self.token_embed_dim, st_embed_dim, bias=True) # projs VAE latents to transformer dim
        # self.z_proj_ln = nn.LayerNorm(st_embed_dim)

        # special token [BUF] for scratch space, row of these added onto each frame
        self.buf_token = nn.Parameter(torch.zeros(1, st_embed_dim))

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
        self.scale_factor = 0.07843137255

        # ----- intialize the vae -----
        if vae_config:
            self.instantiate_vae(vae_config)
            assert isinstance(self.vae, AutoencoderKL)
            self.vae_embed_dim = self.vae.latent_dim
            assert self.vae.latent_dim == vae_embed_dim
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
        torch.nn.init.kaiming_normal_(self.buf_token)
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
    
    def shuffled_token_indices(self, batch_size):
        # contains B randomized vectors of indices going from 0, 1, ..., HW-1
        indices = []

        for _ in range(batch_size):
            order = np.array(list(range(self.frame_seq_len)))  # (HW)
            np.random.shuffle(order)                           # (HW)
            indices.append(order)

        indices = torch.Tensor(np.array(indices)).to(self.device).long()  # (B, HW)
        return indices
    
    def shuffled_token_indices(self, batch_size):
        HW = self.frame_seq_len
        perms = [torch.randperm(HW, device=self.device) for _ in range(batch_size)]  # [(HW), ..., (HW)]
        perms = torch.stack(perms, dim=0)                                            # (B, HW)
        return perms  # (B, HW)

    def random_masking(self, x, masking_rate=None, custom_orders=None, prev_masking=True):
        B, T, H, W, D = x.shape
        P = self.num_prev_frames
        HW = self.frame_seq_len
        assert HW == H*W

        # determine how many tokens should me masked in each non-memory context frame
        pred_mask_rate = self.mask_ratio_generator.rvs(1)[0] if not masking_rate else masking_rate
        num_masked_pred = int(np.floor(HW * pred_mask_rate))
        num_masked_ctx = int(num_masked_pred * self.prev_masking_rate)

        # random mask for the frame to be predicted (by default, final frame at index 0)
        mask = torch.zeros((B, HW), dtype=bool, device=x.device)  # (B, HW)
        pred_orders = self.shuffled_token_indices(B) if custom_orders is None else custom_orders
        pred_mask = torch.scatter(
            mask,
            dim=-1, 
            index=pred_orders[:, :num_masked_pred], 
            src=torch.ones(B, HW, dtype=bool, device=x.device)
        )  # (B, HW)
        
        # random masks for previous non-memory context frames (attention-masks not MAR masks for diffused tokens)
        if prev_masking:
            ctx_masks = torch.zeros((B, P-1, HW), dtype=torch.bool, device=x.device)  # (B, P-1, HW)
            for p in range(P-1):
                ctx_orders = self.shuffled_token_indices(B)
                ctx_masks[:, p, :] = torch.zeros((B, HW), dtype=torch.bool, device=x.device).scatter(
                    dim=-1,
                    index=ctx_orders[:, :num_masked_ctx],
                    src=torch.ones((B, HW), dtype=torch.bool, device=x.device)
                )
        else:
            ctx_masks = None

        return pred_mask, ctx_masks  # (B, HW), (B, P-1, HW)

    def add_pose_action_timestamp_embeddings(self, x, poses, timestamps, actions, batch_nframes, is_decoder=False):
        B, T, H, W, D = x.shape
        
        if is_decoder:
            H -= 1

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
        timestamp_embeddings = timestamp_embeddings.view(B, T, 1, 1, D)                  # (B, T, 1, 1, D)
        timestamp_embeddings = timestamp_embeddings.expand(-1, -1, H, W, -1)             # (B, T, H, W, D)
        # zero out timestamp embeddings for padding frames
        frame_idx = torch.arange(T, device=x.device).view(1, T, 1, 1, 1)                 # (1, T, 1, 1, 1)
        valid = frame_idx < batch_nframes.view(B, 1, 1, 1, 1)                            # (B, T, 1, 1, 1)
        timestamp_embeddings = timestamp_embeddings * valid
        assert timestamp_embeddings.shape == (B, T, H, W, D)                             # (B, T, H, W, D)

        # construct action embeddings matching latent shape
        action_embeddings = self.action_embedder(actions)                   # (B, D)
        action_embeddings = rearrange(action_embeddings, 'b d -> b 1 1 d')  # add singleton dims for H and W

        # return the latent with the embeddings added
        x[:, 0, :H, :, :] += action_embeddings  # only final pred frame gets action embeddings
        x[:, :, :H, :, :] += pose_embeddings + timestamp_embeddings   

        return x  # (B, T, H or H+1, W, D)

    def add_buffer_tokens(self, x):
        B, T, H, W, D = x.shape

        # construct [BUF] token buffer (expected to be a slice varying along time-dimension)
        token_buffer = self.buf_token.view(1, 1, 1, 1, D).repeat(B, T, 1, W, 1)  # (B, T, 1, W, D)

        # concat [BUF] token buffer
        # these concatenations are happening along H, the height dimension
        x = torch.cat([x, token_buffer], dim=-3)  # dim=-3 is H

        assert x.shape == (B, T, H+1, W, D)
        return x  # (B, T, H+1, W, D)

    def forward_encoder(self, x, actions, poses, timestamps, batch_nframes, s_attn_mask=None, t_attn_mask=None):
        # x:       (B, T, H, W, token_embed_dim)
        # actions: (B, 25)
        # poses:   (B, T, 40H, 40W, 6)

        # project and layer normalize
        x = self.z_proj(x)  # (B, T, H, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_action_timestamp_embeddings(x, poses, timestamps, actions, batch_nframes)  # (B, T, H, W, D)

        # add token buffers for prediction and context (on memory frames)
        x = self.add_buffer_tokens(x)  # (B, T, H+1, W, D)

        # pass through each encoder spatio-temporal attention block
        for block in self.encoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)  # (B, T, H+1, W, D)
        
        # final layer norm before passing to the decoder
        x = self.encoder_norm(x)  # (B, T, H+1, W, D)

        return x  # (B, T, H+1, W, D)

    def forward_decoder(self, x, actions, poses, timestamps, mask, batch_nframes, pred_idx=0, s_attn_mask=None, t_attn_mask=None):
        # x:       (B, T, H+1, W, D)
        # mask:    (B, (H+1)W)
        B, T, H, W, D = x.shape
        H -= 1  # encoder would have concatenated token buffer onto x's H dim

        # when intersected, t_mask selects the predicted frame, and s_mask selects the tokens within that frame
        s_mask = mask.view(B, 1, H+1, W)                                             # (B, 1, H+1, W)
        t_mask = (torch.arange(T, device=self.device) == pred_idx).view(1, T, 1, 1)  # (1, T, 1, 1)
        full_mask = s_mask & t_mask                                                  # (B, T, H+1, W)

        # replace ground-truth latent tokens with [MASK] tokens based on full_mask
        x = torch.where(full_mask.unsqueeze(-1), self.mask_token, x)                 # (B, T, H+1, W, D)

        # add pose and timestamp embeddings
        x = self.add_pose_action_timestamp_embeddings(x, poses, timestamps, actions, batch_nframes, is_decoder=True)  # (B, T, H+1, W, D)

        # pass through each decoder spatio-temporal attention block
        for block in self.decoder_blocks:
            x = block(x, s_attn_mask=s_attn_mask, t_attn_mask=t_attn_mask)           # (B, T, H+1, W, D)

        # final layer norm before passing to the diffusion backbone
        x = self.decoder_norm(x)                                                     # (B, T, H+1, W, D)

        # remove the extra token buffer that was concatenated by the encoder onto x's H dim
        x = x[:, :, :H, :, :]                                                        # (B, T, H, W, D)

        return x  # (B, T, H, W, D)
    
    def forward_diffusion(self, z, target, mask):
        target = rearrange(target, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)  # (4BHW, D)
        z = z + self.diffusion_pos_emb_learned
        z = rearrange(z, "b h w d -> (b h w) d").repeat(self.diffusion_batch_mul, 1)            # (4BHW, D)
        mask = rearrange(mask, "b s -> (b s)").repeat(self.diffusion_batch_mul)                 # (4BHW)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def construct_attn_masks(self, x, pred_mask, pred_idx=0, ctx_mask=None, padding_mask=None):
        B, T, H, W, D = x.shape
        P = self.num_prev_frames

        if ctx_mask is not None:
            ctx_mask_left = ctx_mask[:, :pred_idx, :]
            ctx_mask_middle = torch.zeros_like(pred_mask, dtype=torch.bool).unsqueeze(-2)
            ctx_mask_right = ctx_mask[:, pred_idx:, :]
            ctx_mask = torch.cat([ctx_mask_left, ctx_mask_middle, ctx_mask_right], dim=-2)  # (B, P, (H+1)W)

        # --- spatial attn masks ---
        valid_hw = torch.ones(B, T, (H+1)*W, dtype=torch.bool, device=self.device)  # (B, T, (H+1)W)

        if padding_mask is not None:
            # mask out entire frames if they were added as padding (padding_mask is (B, T),
            # and padding_mask[:, t] = 0 if the frame at timestep t is padding)
            valid_hw &= padding_mask.unsqueeze(-1)  # (B, T, (H+1)W)

        if self.training and ctx_mask is not None:
            # mask random tokens of previous non-memory context frames (i.e. set to 0)
            valid_hw[:, :P, :] = ~ctx_mask  # (B, T, (H+1)W)

        # construct the decoder spatial attention mask
        # --------------------------------------------*
        # valid_hw.unsqueeze(-1) is (B, T, (H+1)W, 1) |
        # valid_hw.unsqueeze(-2) is (B, T, 1, (H+1)W) |
        # --------------------------------------------*
        # this outer-product logical AND operation broadcasts both into (B, T, (H+1)W, (H+1)W)
        # so that s_attn_mak[b, t, i, j] = 1 iff tokens i and j are individually valid within frame t
        s_attn_mask_dec = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        # attention mechanism expects (B, num_heads, sequence_len, sequence_len) mask, but all heads have same mask
        s_attn_mask_dec = rearrange(s_attn_mask_dec, "b t hw1 hw2 -> (b t) 1 hw1 hw2")
        
        # construct the encoder spatial attention mask in a similar fashion
        # but prevent any masked tokens on the predicted frame from being seen
        valid_hw[:, pred_idx, :] = ~pred_mask
        s_attn_mask_enc = valid_hw.unsqueeze(-1) & valid_hw.unsqueeze(-2)
        s_attn_mask_enc = rearrange(s_attn_mask_enc, "b t hw1 hw2 -> (b t) 1 hw1 hw2")

        # --- temporal attn mask ---
        L = (H+1)*W
        valid_t = torch.ones(B, L, T, dtype=torch.bool, device=self.device)                # (B, (H+1)W, T)
        batch_idx = torch.arange(B, device=valid_t.device).view(B, 1, 1).expand(-1, P, L)  # (B, P, (H+1)W)
        time_idx = torch.arange(P, device=valid_t.device).view(1, P, 1).expand(B, -1, L)   # (B, P, (H+1)W)
        len_idx = torch.arange(L, device=valid_t.device).view(1, 1, L).expand(B, P, -1)    # (B, P, (H+1)W)

        if padding_mask is not None:
            # mask out entire frames if they were added as padding (padding_mask is (B, T),
            # and padding_mask[:, t] = 0 if the frame at timestep t is padding)
            valid_t &= padding_mask.unsqueeze(-2)  # (B, (H+1)W, T)

        if self.training and ctx_mask is not None:
            # mask random tokens of previous non-memory context frames (i.e. set to 0)
            valid_t[batch_idx[ctx_mask], len_idx[ctx_mask], time_idx[ctx_mask]] = False

        # construct the decoder temporal attention mask using the outer-product logical AND
        t_attn_mask_dec = valid_t.unsqueeze(-1) & valid_t.unsqueeze(-2)
        t_attn_mask_dec = rearrange(t_attn_mask_dec, "b hw t1 t2 -> (b hw) 1 t1 t2")

        # construct the encoder temporal attention mask in a similar fashion
        # but prevent any masked tokens on the predicted frame from being seen
        valid_t[:, :, pred_idx] = ~pred_mask
        t_attn_mask_enc = valid_t.unsqueeze(-1) & valid_t.unsqueeze(2)
        t_attn_mask_enc = rearrange(t_attn_mask_enc, "b hw t1 t2 -> (b hw) 1 t1 t2")

        return s_attn_mask_enc, t_attn_mask_enc, s_attn_mask_dec, t_attn_mask_dec

    def masked_encoder_decoder(self, x, actions, poses, timestamps, 
                               pred_mask, ctx_mask, batch_nframes, 
                               pred_idx=0, padding_mask=None):
        B = x.shape[0]

        # cache gt latents
        z_gt = x.clone().detach()

        # generate masks for all frames in previous frame window
        B, T, H, W, D = x.shape
        P = self.num_prev_frames

        # add padding along the H dimension (since we concat token buffers for the encoder-decoder to use)
        padded_pred_mask = rearrange(pred_mask, "b (h w) -> b h w", h=H)                                                        # (B, H, W)
        padded_pred_mask = torch.cat([padded_pred_mask, torch.zeros((B, 1, W), dtype=torch.bool, device=self.device)], dim=-2)  # (B, H+1, W)
        padded_pred_mask = rearrange(padded_pred_mask, "b h w -> b (h w)")                                                      # (B, (H+1)W)

        if ctx_mask is not None:
            padded_ctx_mask = rearrange(ctx_mask, "b t (h w) -> b t h w", t=P-1, h=H)                                                # (B, P-1, H, W)
            padded_ctx_mask = torch.cat([padded_ctx_mask, torch.zeros(B, P-1, 1, W, dtype=torch.bool, device=self.device)], dim=-2)  # (B, P-1, H+1, W)
            padded_ctx_mask = rearrange(padded_ctx_mask, "b t h w -> b t (h w)")                                                     # (B, P-1, (H+1)W)
        else:
            padded_ctx_mask = None

        # construct spatial and temporal attention masks and pass through encoder and decoder
        (s_attn_mask_enc, t_attn_mask_enc, s_attn_mask_dec, t_attn_mask_dec) = self.construct_attn_masks(
            x, padded_pred_mask, pred_idx=pred_idx, ctx_mask=padded_ctx_mask, padding_mask=padding_mask
        )
        z = self.forward_encoder(
            x, actions, poses, timestamps, batch_nframes, 
            s_attn_mask=s_attn_mask_enc, t_attn_mask=t_attn_mask_enc
        )
        z = self.forward_decoder(
            z, actions, poses, timestamps, padded_pred_mask, batch_nframes, 
            pred_idx=pred_idx, s_attn_mask=s_attn_mask_dec, t_attn_mask=t_attn_mask_dec
        )
        return z_gt, z  # (B, T, H, W, D), (B, T, H, W, D)

    def forward(self, x, actions, poses, timestamps, batch_nframes, pred_idx=0, padding_mask=None):
        # scale the input tensor x to a standard normal distribution
        B = x.shape[0]
        x = x * self.scale_factor

        # patchify latents
        x = rearrange(x, "b t s c -> (b t) s c")
        x = self.patchify(x) # (bt) h w d (different h and w bc of patchifying)
        x = rearrange(x, "(b t) h w d -> b t h w d", b=B)
        
        # pass through the main masked spatio-temporal attention mechanism
        pred_mask, ctx_mask = self.random_masking(x, masking_rate=None)  # (B, HW), (B, P-1, HW)
        z_gt, z = self.masked_encoder_decoder(x, actions, poses, timestamps, pred_mask, ctx_mask, batch_nframes, pred_idx, padding_mask)

        # split into target frame + diffuse
        z_t = z[:, pred_idx, :, :, :]        # (B, H, W, D)
        z_gt_t = z_gt[:, pred_idx, :, :, :]  # (B, H, W, D)
        loss = self.forward_diffusion(z_t, z_gt_t, pred_mask)        
        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()
        opt.zero_grad()

        # --- parse batch ---
        # assume the layout is [PRED_FRAME, PREV_FRAME, CTX_FRAMES ...]
        frames = batch["frames"].to(self.device) # shape [B, T, L, D]
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
        loss = self(frames, actions, poses, timestamps, batch_nframes, padding_mask=padding_mask)
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
        loss = self(frames, actions, poses, timestamps, batch_nframes, padding_mask=padding_mask)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate)
        warmup_sched = LinearLR(
            optim,
            start_factor=4e-1,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        # cosine_sched = CosineAnnealingLR(
        #     optim,
        #     T_max=self.warmup_steps * 5, # warmup steps ~ batches / epoch
        #     eta_min=0.0
        # )
        scheduler = SequentialLR(
            optim,
            schedulers=[warmup_sched],
            milestones=[]
            # schedulers=[warmup_sched, cosine_sched],
            # milestones=[self.warmup_steps]
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def sample(self, x, actions, poses, timestamps, batch_nframes, num_mar_iters=4, pred_idx=0, prev_masking=False):
        B, L = len(x), self.num_frames * self.frame_seq_len

        # scale the input tensor x to match a standard normal distribution
        x = x * self.scale_factor

        # patchify latents
        x = rearrange(x, "b t s c -> (b t) s c")
        x = self.patchify(x) # (bt) h w d (different h and w bc of patchifying)
        x = rearrange(x, "(b t) h w d -> b t h w d", b=B)

        # construct padding mask
        idx = torch.arange(self.num_frames, device=self.device).expand(B, self.num_frames)
        padding_mask = idx < batch_nframes.unsqueeze(1) # b t

        # gen init mask
        orders = self.shuffled_token_indices(B)
        pred_mask, ctx_mask = self.random_masking(x, masking_rate=1.0, custom_orders=orders, prev_masking=prev_masking)

        # store for later logging (see MaskedImageLogger class)
        self._last_pred_idx = pred_idx
        self._last_pred_mask = pred_mask.detach().cpu()
        self._last_ctx_mask = (ctx_mask.detach().cpu() if ctx_mask is not None else None)
        self._last_pred_masks_iters = []
        self._last_pred_iters = []
        
        for step in range(num_mar_iters):
            # get prediction with cur state of masks
            _, z = self.masked_encoder_decoder(x, actions, poses, timestamps, pred_mask, ctx_mask, batch_nframes, pred_idx, padding_mask)

            # decide on next masking rate, dictating which we actually predict here
            masking_rate = np.cos(math.pi / 2. * (step + 1) / num_mar_iters)
            next_pred_mask, _ = self.random_masking(x, masking_rate=masking_rate, custom_orders=orders, prev_masking=prev_masking)
            # what we're actually predicting now -- the exclusion of the next mask and cur mask
            cur_pred_mask = torch.logical_xor(pred_mask, next_pred_mask)

            # do preds
            z = z[:, pred_idx, :, :, :]
            z = z + self.diffusion_pos_emb_learned
            z = rearrange(z, "b h w d -> (b h w) d", b=B, h=self.seq_h, w=self.seq_w)
            z_masked = z[cur_pred_mask.flatten()]
            x_pred = self.diffloss.sample_ddim(z_masked)
            # and stuff these into cur x
            x_cur = rearrange(x[:,pred_idx,:,:,:], "b h w c -> (b h w) c")
            x_cur[cur_pred_mask.flatten()] = x_pred
            x[:,pred_idx,:,:,:] = rearrange(x_cur, "(b h w) c -> b h w c", b=B, h=self.seq_h, w=self.seq_w)

            # save current mask and prediction for visualization
            self._last_pred_masks_iters.append(cur_pred_mask.detach().cpu())
            self._last_pred_iters.append(x[:,pred_idx,:,:,:].detach().cpu())

            # update mask for next iter
            pred_mask = next_pred_mask

        x_pred = x[:,pred_idx,:,:,:]
        x_pred = rearrange(x_pred, "b h w d -> b (h w) d")
        x_pred = self.unpatchify(x_pred)

        # undo the scaling operation done to the input tensor (recover the original range of values)
        x_pred = x_pred / self.scale_factor

        return x_pred