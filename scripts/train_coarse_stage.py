import os
import sys

import torch
from audiolm_pytorch import FairseqVQWav2Vec


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_coarse_transformer 
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.trainer import SingleStageTrainer
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = '../audiolm-train/test_audio'

print('loading clap...')


with disable_print():
    clap = create_clap_quantized(device=device, checkpoint_path="./checkpoints/clap-laion-audioset-fusion.pt").to(device)

print('loading wav2vec...')
wav2vec = FairseqVQWav2Vec(
    # checkpoint_path = './hubert/hubert_base_ls960.pt',
    checkpoint_path='./checkpoints/vq-wav2vec_kmeans.pt'
)

print('loading encodec')
encodec_wrapper = create_encodec_24khz(bandwidth=12.).to(device)

# 8 tokens per timestep @ 75 Hz
# lets do 3 coarse 5 fine

print('loading coarse stage...')
coarse_transformer = create_coarse_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    semantic_codebook_size=wav2vec.codebook_size,
    acoustic_codebook_size=encodec_wrapper.codebook_size,
    num_coarse_quantizers=3,
).to(device)

trainer = SingleStageTrainer(
    transformer=coarse_transformer,
    stage='coarse',
    audio_conditioner=clap,
    wav2vec=wav2vec,
    neural_codec=encodec_wrapper,
    folder=audio_folder,
    batch_size=1,
    data_max_seconds=5,
    num_train_steps=1,
    results_folder='./results/coarse',
    accelerate_kwargs={
        'log_with': "tensorboard",
    }
).to(device)

print('training!')
trainer.train()
