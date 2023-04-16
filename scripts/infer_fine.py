'''
1) load audio samples
2) compute clap tokens and coarse tokens
3) run them through fine stage to predict coarse tokens
4) reconstruct audio from coarse + fine tokens
Reconstructed audio should be similar to the original audio if fine stage is working correctly

example usage:

python scripts/infer_fine.py \
    ./data/fma_large/000/000005.mp3 \
    ./data/fma_large/000/000010.mp3 \
    --model_config ./configs/model/musiclm_small.json \
    --fine_path ./results/coarse_continue_1/coarse.transformer.10000.pt
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
                                 create_fine_transformer_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 load_model_config)
from open_musiclm.open_musiclm import (FineStage,
                                       get_or_compute_clap_token_ids,
                                       get_or_compute_acoustic_token_ids)
from open_musiclm.utils import int16_to_float32, float32_to_int16, zero_mean_unit_var_norm
from scripts.train_utils import disable_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference on fine stage')
    parser.add_argument('audio_files', type=str, nargs='+')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json', help='path to model config')
    parser.add_argument('--fine_path', required=True, help='path to fine stage checkpoint')
    parser.add_argument('--temperature', default=0.4, type=float)
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)

    audio_files = args.audio_files
    fine_path = args.fine_path
    rvq_path = args.rvq_path
    seed = args.seed
    temperature = args.temperature

    print(f'running inference on {audio_files}')
    print(f'fine_path: {fine_path}, rvq_path: {rvq_path}, seed: {seed}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, rvq_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)

    print('loading fine stage...')
    fine_transformer = create_fine_transformer_from_config(model_config, fine_path, device)

    fine_stage = FineStage(
        fine_transformer=fine_transformer,
        neural_codec=encodec_wrapper,
        clap=clap
    )

    torch.manual_seed(seed)

    print('loading audio from dataset')

    audios_for_clap = []
    audios_for_encodec = []
    for audio_path in audio_files:
        data, sample_hz = torchaudio.load(audio_path)

        if data.shape[0] > 1:
            data = torch.mean(data, dim=0).unsqueeze(0)

        target_length = int(model_config.global_cfg.fine_audio_length_seconds * sample_hz)

        data = data[:, :target_length]
        audio_for_clap = resample(data, sample_hz, clap.sample_rate)
        audio_for_encodec = resample(data, sample_hz, encodec_wrapper.sample_rate)

        audio_for_clap = int16_to_float32(float32_to_int16(audio_for_clap))
        audio_for_encodec = int16_to_float32(float32_to_int16(audio_for_encodec))

        audios_for_clap.append(audio_for_clap)
        audios_for_encodec.append(audio_for_encodec)

    audios_for_clap = torch.cat(audios_for_clap, dim=0).to(device)
    audios_for_encodec = torch.cat(audios_for_encodec, dim=0).to(device)

    clap_token_ids = get_or_compute_clap_token_ids(None, clap, audios_for_clap, None)
    coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(None, None, audios_for_encodec, encodec_wrapper, model_config.global_cfg.num_coarse_quantizers)

    generated_wave = fine_stage.generate(
        clap_token_ids=clap_token_ids,
        coarse_token_ids=coarse_token_ids,
        max_time_steps=int(model_config.global_cfg.fine_audio_length_seconds * model_config.encodec_cfg.output_hz),
        reconstruct_wave=True,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        temperature=temperature,
    )

    generated_wave = rearrange(generated_wave, 'b n -> b 1 n').detach().cpu()
    for i, wave in enumerate(generated_wave):
        torchaudio.save(f'results/fine_reconstruct_{i}.wav', wave, encodec_wrapper.sample_rate)