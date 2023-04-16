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
    --model_config ./configs/model/musiclm_small.json \
    --coarse_path ./results/coarse_continue_1/coarse.transformer.10000.pt

'''

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
from torchaudio.functional import resample

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_coarse_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 load_model_config)
from open_musiclm.open_musiclm import (CoarseStage,
                                       get_or_compute_clap_token_ids,
                                       get_or_compute_semantic_token_ids)
from open_musiclm.utils import int16_to_float32, float32_to_int16, zero_mean_unit_var_norm
from scripts.train_utils import disable_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference on coarse stage')
    parser.add_argument('audio_files', type=str, nargs='+')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json', help='path to model config')
    parser.add_argument('--coarse_path', required=True, help='path to coarse stage checkpoint')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)

    audio_files = args.audio_files
    coarse_path = args.coarse_path
    kmeans_path = args.kmeans_path
    rvq_path = args.rvq_path
    seed = args.seed

    print(f'running inference on {audio_files}')
    print(f'coarse_path: {coarse_path}, kmeans_path: {kmeans_path}, rvq_path: {rvq_path}, seed: {seed}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

    print('loading wav2vec...')
    wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)

    print('loading coarse stage...')
    coarse_transformer = create_coarse_transformer_from_config(model_config, coarse_path, device)

    coarse_stage = CoarseStage(
        coarse_transformer=coarse_transformer,
        neural_codec=encodec_wrapper,
        wav2vec=wav2vec,
        clap=clap
    )

    torch.manual_seed(args.seed)

    print('loading audio from dataset')

    audios_for_clap = []
    audios_for_wav2vec = []
    for audio_path in audio_files:
        data, sample_hz = torchaudio.load(audio_path)

        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(4 * sample_hz)
        normalized_data = zero_mean_unit_var_norm(data)

        data = data[:, :target_length]
        normalized_data = normalized_data[: , :target_length]
        audio_for_clap = resample(data, sample_hz, clap.sample_rate)
        audio_for_wav2vec = resample(normalized_data, sample_hz, wav2vec.target_sample_hz)

        audio_for_clap = int16_to_float32(float32_to_int16(audio_for_clap))
        audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))

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
        max_time_steps=int(model_config.global_cfg.coarse_audio_length_seconds * 75),
        reconstruct_wave=True,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        temperature=0.95,
    )

    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
    for i, wave in enumerate(generated_wave):
        torchaudio.save(f'results/{i}.wav', wave, encodec_wrapper.sample_rate)