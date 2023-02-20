import os
import sys

import torch


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.trainer import ClapRVQTrainer
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = './data/fma_large'

clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
rvq_checkpoint = None   # './checkpoints/clap.rvq.950.pt'
with disable_print():
    clap = create_clap_quantized(device=device, learn_rvq=True, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=rvq_checkpoint).to(device)

trainer = ClapRVQTrainer(
    num_train_steps=1000, 
    batch_size=64,
    accumulate_initial_batch=2,
    audio_conditioner=clap,
    folder=audio_folder,
    results_folder='./results/clap_rvq',
    save_model_every=50,
    save_results_every=25
).to(device)

trainer.train()