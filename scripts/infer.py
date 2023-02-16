import os
import sys
from pathlib import Path

import torch
import torchaudio
from audiolm_pytorch import HubertWithKmeans
from einops import rearrange


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_coarse_transformer, create_fine_transformer, create_semantic_transformer, MusicLM
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.trainer import SingleStageTrainer
from scripts.train_utils import disable_print, get_wav2vec

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = '../audiolm-train/audio'

print('loading clap...')
clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
rvq_checkpoint = './results/semantic/semantic.conditioner_rvq.6000.pt'
with disable_print():
    clap = create_clap_quantized(device=device, learn_rvq=False, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=rvq_checkpoint).to(device)

print('loading wav2vec...')
wav2vec = get_wav2vec(device=device)

print('loading encodec')
encodec_wrapper = create_encodec_24khz(bandwidth=12.).to(device)

semantic_transformer = create_semantic_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    semantic_codebook_size=wav2vec.codebook_size,
).to(device)

print('creating transformers')
coarse_transformer = create_coarse_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    semantic_codebook_size=wav2vec.codebook_size,
    acoustic_codebook_size=encodec_wrapper.codebook_size,
    num_coarse_quantizers=3,
).to(device)

fine_transformer = create_fine_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=clap.codebook_size,
    acoustic_codebook_size=encodec_wrapper.codebook_size,
    num_coarse_quantizers=3,
    num_fine_quantizers=5,
).to(device)

def load_model(model, path):
    path = Path(path)
    assert path.exists(), f'checkpoint does not exist at {str(path)}'
    pkg = torch.load(str(path))
    model.load_state_dict(pkg)

print('loading semantic stage...')
load_model(semantic_transformer, './results/semantic/semantic.transformer.6000.pt')

print('loading coarse stage...')
load_model(coarse_transformer, './results/coarse/coarse.transformer.4000.pt')

print('loading fine stage...')
load_model(fine_transformer, './results/fine/fine.transformer.3000.pt')

print('loading musiclm')
musiclm = MusicLM(
    wav2vec=wav2vec,
    clap=clap,
    neural_codec=encodec_wrapper,
    semantic_transformer=semantic_transformer,
    coarse_transformer=coarse_transformer,
    fine_transformer=fine_transformer,
    unique_consecutive=True,
)

print('generating...')
generated_wave = musiclm.forward(text=['chirping of birds and the distant echos of bells', 'cat meowing'])

print(generated_wave.shape)

generated_wave = rearrange(generated_wave, 'b n -> b 1 n')
for i, wave in enumerate(generated_wave):
    torchaudio.save(f'results/gen_{i}.wav', wave, encodec_wrapper.sample_rate)

