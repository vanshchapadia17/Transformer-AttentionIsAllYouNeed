import torch
from data_loader import decode, vocab_size, device
from model       import GPTLanguageModel

model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))