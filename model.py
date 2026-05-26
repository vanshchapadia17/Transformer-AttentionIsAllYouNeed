import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Hyperparameters ──────────────────────────────────────────
batch_size    = 64      # sequences processed in parallel
block_size    = 256     # maximum context length
max_iters     = 5000    # total training steps
eval_interval = 500     # evaluate every N steps
learning_rate = 3e-4
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200
n_embd        = 384     # embedding dimension
n_head        = 6       # number of attention heads
n_layer       = 6       # number of transformer blocks
dropout       = 0.2     # dropout rate

torch.manual_seed(1337)

# ── Data Loading ─────────────────────────────────────────────
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars      = sorted(list(set(text)))
vocab_size = len(chars)

stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n         = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── Batch Sampler ─────────────────────────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i+block_size]   for i in ix])
    y    = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ── Loss Estimator ────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

# ── 1. Single Attention Head ──────────────────────────────────
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter, so register as buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities")
        # scale by 1/sqrt(head_size) to keep variance stable
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T)

        # Mask future tokens (decoder-style causal attention)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)   # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v   = self.value(x)            # (B, T, head_size)
        out = wei @ v                  # (B, T, head_size)
        return out


# ── 2. Multi-Head Attention ───────────────────────────────────
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)   # projection back to residual stream
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel and concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out


# ── 3. Feed-Forward Network ───────────────────────────────────
class FeedForward(nn.Module):
    """Simple two-layer MLP with ReLU, applied position-wise."""

    def __init__(self, n_embd):
        super().__init__()
        # Inner dimension is 4x n_embd (same as original paper)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── 4. Transformer Block ──────────────────────────────────────
class Block(nn.Module):
    """One full transformer block: attention + feed-forward, both with residual connections."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)  # self-attention
        self.ffwd  = FeedForward(n_embd)                    # feed-forward
        self.ln1   = nn.LayerNorm(n_embd)                   # layer norm before attention
        self.ln2   = nn.LayerNorm(n_embd)                   # layer norm before FFN

    def forward(self, x):
        # Residual connections: x + sublayer(LayerNorm(x))
        # Note: Pre-norm formulation (differs slightly from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────
# FULL GPT MODEL
# ─────────────────────────────────────────────────────────────
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token embedding: maps each token id → n_embd vector
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # Position embedding: maps each position 0..block_size-1 → n_embd vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of N transformer blocks
        self.blocks   = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final layer norm
        self.ln_f     = nn.LayerNorm(n_embd)
        # Language model head: projects n_embd → vocab_size (logits)
        self.lm_head  = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embedding_table(idx)                          # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                              # (B, T, n_embd)

        # Pass through all transformer blocks
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)

        # Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context to block_size (can't exceed position embedding size)
            idx_cond      = idx[:, -block_size:]
            logits, loss  = self(idx_cond)
            logits        = logits[:, -1, :]          # last time step → (B, C)
            probs         = F.softmax(logits, dim=-1) # (B, C)
            idx_next      = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx           = torch.cat((idx, idx_next), dim=1)        # (B, T+1)
        return idx


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
model     = GPTLanguageModel()
m         = model.to(device)

# Print number of parameters
print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Evaluate loss every eval_interval steps
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample batch and train
    xb, yb        = get_batch('train')
    logits, loss  = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ─────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


torch.save(model.state_dict(), 'model.pt')
print("Model saved!")