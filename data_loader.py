import torch

# ── Hyperparameters ───────────────────────────────────────────
batch_size = 64
block_size = 128
device     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load data ─────────────────────────────────────────────────
with open('C:\\Users\\DELL\\Desktop\\Transformer\\data\\train.csv', 'r', encoding='utf-8') as f:
    train_text = f.read()

with open('C:\\Users\\DELL\\Desktop\\Transformer\\data\\validation.csv', 'r', encoding='utf-8') as f:
    val_text = f.read()

# ── Build vocabulary from BOTH datasets combined ──────────────
# Important: use both files so no character is missing from vocab
all_text   = train_text + val_text
chars      = sorted(list(set(all_text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")

# ── Tokenization ──────────────────────────────────────────────
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Convert to tensors ────────────────────────────────────────
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data   = torch.tensor(encode(val_text),   dtype=torch.long)

print(f"Train tokens : {len(train_data):,}")
print(f"Val tokens   : {len(val_data):,}")

# ── Batch sampler ─────────────────────────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i+block_size]     for i in ix])
    y    = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y