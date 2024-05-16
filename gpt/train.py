import os
import numpy as np

from model import GPT

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, '../data')

train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'val.bin')

train_ids = np.memmap(train_file, dtype=np.uint16, mode='r')
val_ids = np.memmap(val_file, dtype=np.uint16, mode='r')

gpt = GPT()
gpt(train_ids)