import os
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
args = sys.argv

if len(args) < 

base_dir = os.path.dirname(__file__)
input_file = os.path.join(base_dir, 'shakespeare.txt')
train_file = os.path.join(base_dir, 'train.bin')
val_file = os.path.join(base_dir, 'val.bin')

with open(input_file, 'r') as f:
  data = f.read()

  train_split = int(0.9 * len(data))
  train = data[:train_split]
  val = data[train_split:]

  train = enc.encode(train)
  val = enc.encode(val)

  train = np.array(train, dtype=np.uint16)
  val = np.array(val, dtype=np.uint16)

  train.tofile(train_file)
  val.tofile(val_file)
