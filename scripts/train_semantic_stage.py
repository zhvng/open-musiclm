import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_semantic_transformer
from open_musiclm.trainer import SingleStageTrainer
from open_musiclm.hf_hubert_kmeans import get_hubert_kmeans
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = './data/fma_large'

print('loading clap...')

clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
rvq_checkpoint = './checkpoints/clap.rvq.950.pt'
with disable_print():
    clap = create_clap_quantized(device=device, learn_rvq=False, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=rvq_checkpoint).to(device)

print('loading wav2vec...')
wav2vec = get_hubert_kmeans(
    model_name='m-a-p/MERT-v0', 
    kmeans_path='./results/hubert_kmeans/kmeans.joblib'
).to(device) 

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
    batch_size=4,
    grad_accum_every=8,
    data_max_seconds=8,
    num_train_steps=7597 * 5,
    results_folder='./results/semantic',
    accelerate_kwargs={
        'log_with': "tensorboard",
        'logging_dir': './logs/semantic'
    }
).to(device)

print('training!')
trainer.train()
