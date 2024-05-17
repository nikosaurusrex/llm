import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.tkn_embd = nn.Embedding(n_vocab, n_embd)
        # learnable embedding
        self.pos_embd = nn.Embedding(ctx_len, n_embd)

        self.head = nn.Linear(n_embd, n_vocab, False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, t = x.size()

        pos = torch.arange(0, t, device=device)

        tkn_enc = self.tkn_embd(x) # (b, t, n_embd)
        pos_enc = self.pos_embd(pos) # (t, n_embd)

        x = tkn_enc + pos_enc # (b, t, n_embd)

        if y is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = self.head(x[:, [-1], :])
            loss = None

        return logits, loss