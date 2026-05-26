import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from data_loader import block_size, device

# ── What this does ────────────────────────────────────────────
# Transformers need two types of embeddings:
#   1. Token embedding   → what is this character?
#   2. Position embedding → where is this character in the sequence?
# Both are learned during training and added together.




#this create a matric of size vocab_size * n_embd
class embeddings(nn.Module):

    def __init__(self, vocab_size, n_embd):  #each character is represented as a vector of size n_embd measn total 66 chracter
        super().__init__()

        # Token embedding: maps each token id → n_embd vector
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        
        # Position embedding: maps each position 0..block_size-1 → n_embd vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)


#B=batch size, T=sequence length, n_embd=embedding dimension
    def forward(self, idx):  
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)

        return tok_emb + pos_emb   # (B, T, n_embd)
    
