import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch.nn as nn
from components.attention import MultiHeadAttention
from components.feedforward import FeedForward

# ── What this does ────────────────────────────────────────────
# One full transformer block stacks:
#   1. Multi-head self-attention   (communication between tokens)
#   2. Feed-forward network        (each token thinks individually)
#
# Both use:
#   - LayerNorm  → stabilizes training (applied BEFORE each sublayer)
#   - Residual connections → x + sublayer(x) prevents vanishing gradients


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)   # before attention
        self.ln2  = nn.LayerNorm(n_embd)   # before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # attention  + residual
        x = x + self.ffwd(self.ln2(x))  # FFN        + residual
        return x