import os
import sys

import torch
import torchaudio
from torchaudio.functional import resample
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from open_musiclm.data import SoundDataset, get_dataloader, get_masked_dataloader


folder = './data/fma_large/000'
dataset = SoundDataset(
    folder,
    max_length_seconds=(None, 1),
    normalize=(True, False),
    target_sample_hz=(16000, 24000),
    seq_len_multiple_of=None,
    ignore_load_errors=True
)

dl = get_masked_dataloader(dataset, batch_size=4, shuffle=False)
dl_iter = iter(dl)

test_steps = 10
for i in range(test_steps):
    batch = next(dl_iter)
    print(batch)

