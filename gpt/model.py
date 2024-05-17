import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

# Self-Attention block
class GPTAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.l_attn = nn.Linear(n_embd, n_embd * 3, bias=False) # * 3 because for query, key and value
        self.l_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.size()

        q, k, v = self.l_attn(x).split(n_embd, dim=2)

        drpt = dropout if self.training else 0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drpt, is_causal=True)

        x = x.transpose(1, 2).contiguous().view(b, t, c) # concat heads

        x = self.l_proj(x)
        x = self.drop(x)

        return x

class GPTFeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(n_embd * 4, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        x = self.drop(x)

        return x
        
class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(n_embd, bias=False)
        self.attention = GPTAttention()
        self.layer_norm2 = nn.LayerNorm(n_embd, bias=False)
        self.ffd = GPTFeedForward()

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffd(self.layer_norm2(x))

        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.tkn_embd = nn.Embedding(n_vocab, n_embd)
        # learnable embedding
        self.pos_embd = nn.Embedding(ctx_len, n_embd)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock() for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, t = x.size()

        pos = torch.arange(0, t, device=device)

        tkn_enc = self.tkn_embd(x) # (b, t, n_embd)
        pos_enc = self.pos_embd(pos) # (t, n_embd)

        x = tkn_enc + pos_enc # (b, t, n_embd)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.layer_norm(x)

        if y is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = self.head(x[:, [-1], :])
            loss = None

        return logits, loss

    def generate(self, x, max_new_tokens, temperature, top_k):
        for _ in range(max_new_tokens):
            idx_cond = x if x.size(1) <= ctx_len else x[:, -ctx_len:]
            logits, _ = self(idx_cond, None)

            logits = logits[:, -1, :] / temperature
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, idx_next), dim=1)

        return x

    def get_parameters(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        return optim_groups
