"""
Adapted from: https://github.com/etched-ai/open-oasis/blob/master/attention.py
Originally based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

import torch
from torch import nn
from torch.nn import functional as F

class TemporalAxialAttention(nn.Module):
