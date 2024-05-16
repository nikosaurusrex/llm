import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

n_vocab = 50304 # rounded up from gpt2
n_embd = 128
ctx_len = 16

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.tkn_embd = nn.Embedding(n_vocab, n_embd)
        # learnable embedding
        self.pos_embd = nn.Embedding(ctx_len, n_embd)

    def forward(self, ids):
        return ids