import torch
from gpt import GPTLanguageModel, decode, device, block_size

# Load the saved model
model = GPTLanguageModel()
model.load_state_dict(torch.load('model.pt', map_location=device))
model.to(device)
model.eval()

# Generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))