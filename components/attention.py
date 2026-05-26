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
# B = batch size
# T = sequence length
# C = embedding dimension
#
# score = softmax(Q @ K.T / sqrt(head_size)) @ V



class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self,n_embd ,head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities")
        # scale by 1/sqrt(head_size) to keep variance stable
        weight  = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)   # (B, T, T)

        # Mask future tokens (decoder-style causal attention)
        weight= weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weight = F.softmax(weight, dim=-1)   # (B, T, T)
        weight = self.dropout(weight)

        v = self.value(x)  # (B,T,head_size)
        out = weight @ v      # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention prpocess parallel."""

    def __init__(self,n_embd,num_heads,head_size,dropout):
        super().__init__()
        self.heads   = nn.ModuleList([
            Head(n_embd, head_size, dropout) for _ in range(num_heads)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head produces (B, T, head_size)
        # Concatenating along last dim gives (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
