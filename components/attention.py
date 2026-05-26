#attention is all you need

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
from torch.nn import functional as F
from data_loader import block_size


# ── What this does ────────────────────────────────────────────
# Self-attention lets every token "look at" every other token
# that came before it and decide how much to attend to each one.
#
# Q (Query)  = what am I looking for?
# K (Key)    = what do I contain?
# V (Value)  = what do I actually share if attended to?
#
# score = softmax(Q @ K.T / sqrt(head_size)) @ V
