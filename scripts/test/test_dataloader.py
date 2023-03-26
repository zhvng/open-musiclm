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

from open_musiclm.data import SoundDataset, SoundDatasetForPreprocessing, PreprocessedDataset, get_dataloader, get_sound_preprocessing_dataloader, get_preprocessed_dataloader


folder = './data/fma_large/000'

# test random crop

dataset = SoundDataset(
    folder,
    max_length_seconds=(1, 5),
    normalize=(True, False),
    target_sample_hz=(16000, 24000),
    seq_len_multiple_of=None,
    ignore_load_errors=True
)

dl = get_dataloader(dataset, batch_size=4, shuffle=False)
dl_iter = iter(dl)

test_steps = 2
for i in range(test_steps):
    batch = next(dl_iter)
    # print(batch)
    for e in batch:
        print(e.shape)

# test preprocessing

dataset = SoundDatasetForPreprocessing(
    folder,
    max_length_seconds=(None, 1),
    normalize=(True, False),
    target_sample_hz=(16000, 24000),
    seq_len_multiple_of=None,
    ignore_load_errors=True
)

dl = get_sound_preprocessing_dataloader(dataset, shuffle=False)
dl_iter = iter(dl)

test_steps = 2
for i in range(test_steps):
    batch = next(dl_iter)
    print(batch)

# # test preprocessed
dataset = PreprocessedDataset(
    './data/fma_preprocessed',
    stage='coarse',
    semantic_window_seconds=10,
    coarse_window_seconds=4,
    fine_window_seconds=2,
    semantic_steps_per_second=50,
    acoustic_steps_per_second=75,
)

dl = get_preprocessed_dataloader(dataset, batch_size=4, shuffle=True)
dl_iter = iter(dl)

test_steps = 2
for i in range(test_steps):
    batch = next(dl_iter)
    for d in batch:
        print(d.shape)
