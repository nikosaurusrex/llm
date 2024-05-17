import os

import torch
import torch.amp

n_vocab = 50304 # rounded up from gpt2
n_layer = 6
n_head = 12
n_embd = 128
ctx_len = 64
batch_size = 32

dropout = 0.0 

learning_rate = 6e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

training_iter = 10000
eval_interval = 1000 # how many steps until I run the eval again
eval_iters = 100 # how many losses are calculated when evaluating
log_interval = 100 # how many steps between each log message

device = 'cuda'
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch_ctx = torch.amp.autocast(device_type=device, dtype=dtype)

compile = False
wandb_log = False

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, '../data')
out_dir = os.path.join(base_dir, '../out')

os.makedirs(out_dir, exist_ok=True)

def pack_config() -> dict:
    return {
        'n_vocab': n_vocab,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'ctx_len': ctx_len,
        'batch_size': batch_size
    }

def unpack_config(c):
    global n_vocab, n_embd, ctx_len, batch_size

    n_vocab = c['n_vocab']
    n_layer = c['n_layer']
    n_head = c['n_head']
    n_embd = c['n_embd']
    ctx_len = c['ctx_len']
    batch_size = c['batch_size']