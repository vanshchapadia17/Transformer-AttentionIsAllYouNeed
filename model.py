import torch
import torch.nn as nn
from torch.nn import functional as F

from data_loader import block_size, device
from components.embeddings import embeddings
from components.block import Block


# ── What this does ────────────────────────────────────────────
# Assembles all components into the full GPT model:
#   Embeddings → N Transformer Blocks → LayerNorm → LM Head

class GPTLanguageModel(nn.Module):

    def __init__(self,vocab_size,n_embd=128,n_head=4,n_layer=4,dropout=0.2):
        super().__init__()

        self.n_embd = n_embd

        self.token_embedding_table = embeddings(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.head = nn.Linear(n_embd, vocab_size)  # language modeling head
    
    def forward(self, idx, targets=None):

        x      = self.token_embedding_table(idx)  # (B, T, n_embd)
        x      = self.blocks(x)                  # (B, T, n_embd)
        x      = self.ln_f(x)                    # (B, T, n_embd)
        logits = self.head(x)                    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -block_size:]
            logits, _    = self(idx_cond)
            logits       = logits[:, -1, :]
            probs        = F.softmax(logits, dim=-1)
            idx_next     = torch.multinomial(probs, num_samples=1)
            idx          = torch.cat((idx, idx_next), dim=1)
        return idx