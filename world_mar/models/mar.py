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

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss

from modules.utils import instantiate_from_config

class WorldMAR(nn.Module):
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
    ):
    
    # intialize the vae