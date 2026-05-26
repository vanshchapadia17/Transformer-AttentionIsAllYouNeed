import torch
from data_loader import get_batch, vocab_size, device
from model       import GPTLanguageModel

# ── Hyperparameters ───────────────────────────────────────────
max_iters     = 5000
eval_interval = 500
eval_iters    = 50
learning_rate = 1e-3

# ── Build model ───────────────────────────────────────────────
model = GPTLanguageModel(vocab_size).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ── Loss estimator ────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y         = get_batch(split)
            _, loss      = model(X, Y)
            losses[k]    = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ── Training loop ─────────────────────────────────────────────
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb      = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ── Save model ────────────────────────────────────────────────
torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")