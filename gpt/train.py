import os
import sys
import time
import math
import numpy as np
import torch
import wandb
import sentencepiece as spm

from config import *
from model import GPT

if len(sys.argv) <= 1:
  print('usage: train.py <new|resume>')
  sys.exit(1)

mode = sys.argv[1]

save_file = os.path.join(models_dir, 'gpt_ckpt.pt')
if len(sys.argv) > 2:
    save_file = sys.argv[2]

train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'val.bin')

train_ids = np.memmap(train_file, dtype=np.uint16, mode='r')
val_ids = np.memmap(val_file, dtype=np.uint16, mode='r')

def get_batch(of='train'):
  data = train_ids if of == 'train' else val_ids

  ri = torch.randint(len(data) - ctx_len, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i:i+ctx_len]).astype(np.int64)) for i in ri])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+ctx_len]).astype(np.int64)) for i in ri])

  return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

tokenizer_file = os.path.join(models_dir, 'tokenizer.model')
sp = spm.SentencePieceProcessor(model_file=tokenizer_file)
n_vocab = sp.vocab_size()

print(n_vocab)

if mode == 'new':
  gpt = GPT()
  num_params = gpt.get_num_params() / 1e6
  print(f'Creating new model with {num_params:.2f}M parameters')
elif mode == 'resume':
  print(f'Resuming model {save_file}')

  checkpoint = torch.load(save_file, map_location=device)

  unpack_config(checkpoint['config'])
  print(f'layers: {n_layer}')

  gpt = GPT()
  gpt.load_state_dict(checkpoint['model'])
else:
  print('Unknown mode parameter. Try new or continue')
  sys.exit(1)

gpt.to(device)

optimizer = torch.optim.AdamW(gpt.get_parameters(), lr=learning_rate, betas=(beta1,beta2), weight_decay=weight_decay)
if mode == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
  print("Compiling the model")
  gpt = torch.compile(gpt)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def save_checkpoint():
  checkpoint = {
    'model': gpt.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': pack_config()
  }

  print("Saving checkpoint")
  torch.save(checkpoint, save_file)

@torch.no_grad()
def estimate_loss():
  out = {}
  gpt.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      with torch_ctx:
        logits, loss = gpt(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  gpt.train()
  return out['train'], out['val']


decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
  # 1) linear warmup for warmup_iters steps
  if it < warmup_iters:
    return learning_rate * it / warmup_iters
  # 2) if it > lr_decay_iters, return min learning rate
  if it > lr_decay_iters:
    return min_lr
  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)

def train():
  best_loss = 15981239

  start_time = time.time()

  for iter in range(training_iter):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    x, y = get_batch('train')

    with torch_ctx:
      logits, loss = gpt(x, y)
    
    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    if iter % log_interval == 0:
      cur_time = time.time()
      delta_time = cur_time - start_time
      print(f'iter {iter}, loss {loss.item():.2f}, time {delta_time*1000:.2f}ms')

      start_time = cur_time

    if iter % eval_interval == 0:
      train_loss, val_loss = estimate_loss()
      if wandb_log:
        logger.log({
          "iter": iter,
          "train/loss": train_loss,
          "val/loss": val_loss,
          "lr": lr
        })
      else:
        print(f'iter: {iter}, train/loss: {train_loss}, val/loss: {val_loss}, lr: {lr}')

      if val_loss < best_loss:
        save_checkpoint()
        best_loss = val_loss
    
if wandb_log:
  logger = wandb.init(project="llm", name="gpt", config=pack_config())
train()
