from functools import wraps
import logging
import sys
import os
from audiolm_pytorch import HubertWithKmeans

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_wav2vec(device):
    wav2vec = HubertWithKmeans(
        checkpoint_path = './checkpoints/hubert_base_ls960.pt',
        kmeans_path = './checkpoints/hubert_base_ls960_L9_km500.bin',
        target_sample_hz=16000,
    ).to(device)

    return wav2vec
