import sys
import asyncio

import torch
import tiktoken

from config import *
from model import GPT

if len(sys.argv) <= 1:
  print('usage: generate.py <model-file>')
  sys.exit(1)

model_file = sys.argv[1]

gpt = GPT()

checkpoint = torch.load(model_file, map_location=device)
gpt.load_state_dict(checkpoint['model'])
unpack_config(checkpoint['config'])

gpt.eval()
gpt.to(device)

enc = tiktoken.get_encoding('cl100k_base')

if compile:
  gpt = torch.compile(gpt)

start = input("Prompt: ")
ids = enc.encode(start)
x = torch.tensor(ids, device=device, dtype=torch.long)[None, ...]

max_tokens = 1000
temperature = 1.0
top_k = 200 # retain only the top_k most likely tokens, other to 0 probability

async def generate():
  with torch.no_grad():
    with torch_ctx:
      async for token in gpt.generate(x, max_tokens, temperature, top_k):
        print(enc.decode(token[0].tolist()), end="")

asyncio.run(generate())
