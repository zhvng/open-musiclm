import os
import sys

import torch


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.trainer import ClapRVQTrainer
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
# rvq_checkpoint = './results/semantic/semantic.conditioner_rvq.20000.pt'
with disable_print():
    clap = create_clap_quantized(device=device, learn_rvq=True, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=None).to(device)

corrupted_files = ['fma_small/098/098565.mp3',
                   'fma_small/098/098567.mp3',
                   'fma_small/098/098569.mp3',
                   'fma_small/099/099134.mp3',
                   'fma_small/108/108925.mp3',
                   'fma_small/133/133297.mp3']
trainer = ClapRVQTrainer(
    num_train_steps=1000, 
    batch_size=32, 
    audio_conditioner=clap, 
    folder='../audiolm-train/audio', 
    results_folder='./results/clap_rvq',
    ignore_files=corrupted_files,
    save_model_every=100,
    save_results_every=50
).to(device)

trainer.train()