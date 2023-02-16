import os
import sys

import torch
from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_semantic_transformer
from open_musiclm.trainer import SingleStageTrainer
from scripts.train_utils import disable_print, get_wav2vec

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = '../audiolm-train/audio'

print('loading clap...')


with disable_print():
    clap = create_clap_quantized(device=device, learn_rvq=True, checkpoint_path="./checkpoints/clap-laion-audioset-fusion.pt").to(device)

print('loading wav2vec...')
wav2vec = get_wav2vec(device=device)

print('loading semantic stage...')
semantic_transformer = create_semantic_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    semantic_codebook_size=wav2vec.codebook_size,
).to(device)

corrupted_files = ['fma_small/098/098565.mp3',
                   'fma_small/098/098567.mp3',
                   'fma_small/098/098569.mp3',
                   'fma_small/099/099134.mp3',
                   'fma_small/108/108925.mp3',
                   'fma_small/133/133297.mp3']
trainer = SingleStageTrainer(
    transformer=semantic_transformer,
    stage='semantic',
    audio_conditioner=clap,
    wav2vec=wav2vec,
    folder=audio_folder,
    batch_size=2,
    grad_accum_every=4,
    data_max_seconds=8,
    num_train_steps=7597 * 5,
    results_folder='./results/semantic2',
    ignore_files=corrupted_files,
    accelerate_kwargs={
        'log_with': "tensorboard",
        'logging_dir': './logs/semantic'
    }
).to(device)

print('training!')
trainer.train()
