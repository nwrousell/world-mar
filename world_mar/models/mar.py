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

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss

from modules.utils import instantiate_from_config

import pytorch_lightning as pl

class WorldMAR(pl.LightningModule):
    """
    Assumptions Praccho's making:
        vae is an AutoencoderKL, future me can change this to any encoder decoder, but we like LDM's
        so we should be using it
    
    Req'd args... you need to give me a:
        - vae: think that has an eVk

    """
    def __init__(self, 
                 vae_config, # should be an AutoencoderKL 
                 img_height=360, img_width=640,
                 encoder_embed_dim=512, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
                 warmup_steps=1000 # TODO: change this depending on dataset size
    ):
        super().__init__()
        self.automatic_optimization = False

        # intialize the vae
        self.instantiate_from_config(vae_config)
        
        # initialize diff loss
        # TODO: make cutomizable as MLP (per patch?) vs DiT (per frame).
        #       for now, assuming more lightweight MLP
        self.diffloss = DiffLoss
    
    def instantiate_vae(self, vae_config):
        self.vae = instantiate_from_config(vae_config)
        self.vae.train = lambda self, mode=True: self
        for param in self.vae.parameters():
            param.requires_grad=False

    def forward(self, x):
        # TODO: fill this out more detail
        # 1) encode frame
        x = self.vae.encode(...)

        # 2) ma
        x = self

    def training_step(self, batch, batch_idx):
        # TODO: parse batch, whether dict or tuple
        opt = self.optimizers()
        lr_sched = self.lr_schedulers()

        opt.zero_grad()
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
