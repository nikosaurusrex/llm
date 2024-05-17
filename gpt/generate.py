import sys

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

if compile:
    gpt = torch.compile(gpt)

enc = tiktoken.get_encoding("gpt2")

start = input("Prompt: ")
ids = enc.encode(start)
x = torch.tensor(ids, device=device)[None, ...]

max_tokens = 20
temperature = 0.9
top_k = 200 # retain only teh top_k most likely tokens, other to 0 probability

with torch.no_grad():
    with torch_ctx:
        y = gpt.generate(x, max_tokens, temperature, top_k)
        print(enc.decode(y[0].tolist()))