import os
import sys

import torch
from audiolm_pytorch import FairseqVQWav2Vec


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_semantic_transformer
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

print('loading semantic stage...')
semantic_transformer = create_semantic_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    semantic_codebook_size=wav2vec.codebook_size,
).to(device)

trainer = SingleStageTrainer(
    transformer=semantic_transformer,
    stage='semantic',
    audio_conditioner=clap,
    wav2vec=wav2vec,
    folder=audio_folder,
    batch_size=1,
    data_max_seconds=10,
    num_train_steps=1,
    results_folder='./results/semantic',
    accelerate_kwargs={
        'log_with': "tensorboard",
        'logging_dir': './logs/semantic'
    }
).to(device)

print('training!')
trainer.train()
