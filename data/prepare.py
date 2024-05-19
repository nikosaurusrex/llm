import os
import sys
import numpy as np
from datasets import load_dataset
import sentencepiece as spm

args = sys.argv

if len(args) < 3:
  print("Usage: python prepare.py <dataset> <vocab-size>")
  exit(1)

dataset_name = args[1]
vocab_size = int(args[2])

base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, '../models')
train_file = os.path.join(base_dir, 'train.bin')
val_file = os.path.join(base_dir, 'val.bin')
data_file = os.path.join(base_dir, 'data.txt')

data = None
if not os.path.exists(data_file):
  if dataset_name == 'shakespeare':
    input_file = os.path.join(base_dir, 'shakespeare.txt')

    with open(input_file, 'r') as f:
      data = f.read()
  elif dataset_name == 'github':
    ds = load_dataset('codeparrot/github-code', streaming=True, split='train', languages=['C'])
    total_count = 0
    scripts = []

    for data in iter(ds):
      if len(scripts) > 1000:
        break

      lang = data['language']
      if lang == 'C':
        scripts.append(data['code'])

      total_count += 1

    data = ''
    for script in scripts:
      data += script
  else:
    print('unknown dataset')
    exit(1)

  with open(data_file, 'w') as f:
    f.write(data) 
else:
  with open(data_file, 'r') as f:
    data = f.read()
    
if data is not None:
  spm.SentencePieceTrainer.train(input=data_file, model_prefix=f'{models_dir}/tokenizer', vocab_size=vocab_size)

  tokenizer_file = os.path.join(models_dir, 'tokenizer.model')
  sp = spm.SentencePieceProcessor(model_file=tokenizer_file)

  train_split = int(0.9 * len(data))
  train = data[:train_split]
  val = data[train_split:]

  train = sp.encode(train)
  val = sp.encode(val)

  train = np.array(train, dtype=np.uint16)
  val = np.array(val, dtype=np.uint16)

  train.tofile(train_file)
  val.tofile(val_file)
