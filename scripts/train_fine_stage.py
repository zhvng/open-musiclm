import os
import sys

import torch


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_fine_transformer 
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.trainer import SingleStageTrainer
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = '../audiolm-train/test_audio'

print('loading clap...')
with disable_print():
    clap = create_clap_quantized(device=device, checkpoint_path="./checkpoints/clap-laion-audioset-fusion.pt").to(device)

print('loading encodec')
encodec_wrapper = create_encodec_24khz(bandwidth=12.).to(device)

# 8 tokens per timestep @ 75 Hz
# lets do 3 coarse 5 fine

print('loading fine stage...')
coarse_transformer = create_fine_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    acoustic_codebook_size=encodec_wrapper.codebook_size,
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
).to(device)

trainer = SingleStageTrainer(
    transformer=coarse_transformer,
    stage='fine',
    audio_conditioner=clap,
    neural_codec=encodec_wrapper,
    folder=audio_folder,
    batch_size=1,
    data_max_seconds=3,
    num_train_steps=1,
    results_folder='./results/fine',
    accelerate_kwargs={
        'log_with': "tensorboard",
    }
).to(device)

print('training!')
trainer.train()
