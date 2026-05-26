
# ── What this does ────────────────────────────────────────────
# After attention, each token processes its own information
# independently through this small MLP.
# It gives the model capacity to "think" about what it attended to.
#
# Structure: Linear → ReLU → Linear → Dropout
# Inner dimension is 4x n_embd (from the original paper)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)