'''
1) load audio samples
2) compute clap tokens and semantic tokens
3) run them through coarse stage to predict coarse tokens
4) reconstruct audio from coarse tokens
Reconstructed audio should be semantically similar to the original audio if hubert-kmeans and coarse stage are working correctly

example usage:

python scripts/infer_coarse.py \
    ./data/fma_large/000/000005.mp3 \
    ./data/fma_large/000/000010.mp3 \
    --coarse_path ./results/coarse_continue_1/coarse.transformer.10000.pt 

'''

import os
import sys
from pathlib import Path

import torch
import torchaudio
from torchaudio.functional import resample
from einops import rearrange
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_coarse_transformer, get_or_compute_clap_token_ids, get_or_compute_semantic_token_ids, CoarseStage
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.hf_hubert_kmeans import get_hubert_kmeans
from scripts.train_utils import disable_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference on coarse stage')
    parser.add_argument('audio_files', type=str, nargs='+')
    parser.add_argument('--coarse_path', required=True, help='path to coarse stage checkpoint')
    parser.add_argument('--clap_path', default='./checkpoints/clap-laion-audioset-fusion.pt')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    audio_files = args.audio_files
    coarse_path = args.coarse_path
    kmeans_path = args.kmeans_path
    clap_path = args.clap_path
    rvq_path = args.rvq_path
    seed = args.seed

    print(f'running inference on {audio_files}')
    print(f'coarse_path: {coarse_path}, kmeans_path: {kmeans_path}, clap_path: {clap_path}, rvq_path: {rvq_path}, seed: {seed}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    with disable_print():
        clap = create_clap_quantized(device=device, learn_rvq=False, checkpoint_path=clap_path, rvq_checkpoint_path=rvq_path).to(device)

    print('loading wav2vec...')
    wav2vec = get_hubert_kmeans(
        model_name='m-a-p/MERT-v0', 
        kmeans_path=kmeans_path,
        normalize_input=True,
        normalize_embeds=True,
    ).to(device)

    print('loading encodec')
    encodec_wrapper = create_encodec_24khz(bandwidth=6.).to(device)

    print('creating coarse transformer')
    coarse_transformer = create_coarse_transformer(
        dim=1024,
        depth=6,
        clap_codebook_size=clap.codebook_size,
        semantic_codebook_size=wav2vec.codebook_size,
        acoustic_codebook_size=encodec_wrapper.codebook_size,
        num_coarse_quantizers=3,
    ).to(device)

    def load_model(model, path):
        path = Path(path)
        assert path.exists(), f'checkpoint does not exist at {str(path)}'
        pkg = torch.load(str(path))
        model.load_state_dict(pkg)

    print('loading coarse stage...')
    load_model(coarse_transformer, coarse_path)

    coarse_stage = CoarseStage(
        coarse_transformer=coarse_transformer,
        neural_codec=encodec_wrapper,
        wav2vec=wav2vec,
        clap=clap
    )

    torch.manual_seed(99)

    print('loading audio from dataset')
    
    audios_for_clap = []
    audios_for_wav2vec = []
    for audio_path in audio_files:
        data, sample_hz = torchaudio.load(audio_path)

        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(4 * sample_hz)
        data = data[:, :target_length]
        audio_for_clap = resample(data, sample_hz, clap.sample_rate)
        audio_for_wav2vec = resample(data, sample_hz, wav2vec.target_sample_hz)

        audios_for_clap.append(audio_for_clap)
        audios_for_wav2vec.append(audio_for_wav2vec)

    audios_for_clap = torch.cat(audios_for_clap, dim=0).to(device)
    audios_for_wav2vec = torch.cat(audios_for_wav2vec, dim=0).to(device)

    clap_token_ids = get_or_compute_clap_token_ids(None, clap, audios_for_clap, None)
    semantic_token_ids = get_or_compute_semantic_token_ids(None, audios_for_wav2vec, wav2vec)

    generated_wave = coarse_stage.generate(
        clap_token_ids=clap_token_ids,
        semantic_token_ids=semantic_token_ids,
        coarse_token_ids=None,
        max_time_steps=int(4 * 75),
        reconstruct_wave=True,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        temperature=0.95,
    )

    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
    for i, wave in enumerate(generated_wave):
        torchaudio.save(f'results/{i}.wav', wave, encodec_wrapper.sample_rate)