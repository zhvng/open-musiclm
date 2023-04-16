'''
example usage:

python3 scripts/infer.py \
  --semantic_path ./results/semantic/semantic.transformer.10000.pt \
  --coarse_path ./results/coarse/coarse.transformer.10000.pt \
  --fine_path ./results/fine/fine.transformer.10000.pt \
  --model_config ./configs/model/musiclm_small.json \
  --return_coarse_wave
'''

import os
import sys

import torch
import torchaudio
from einops import rearrange
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import load_model_config, create_musiclm_from_config

prompts = [
    [
        'The main soundtrack of an arcade game. It is fast-paced and upbeat, with a catchy electric guitar riff. The music is repetitive and easy to remember, but with unexpected sounds, like cymbal crashes or drum rolls.',
        'A fusion of reggaeton and electronic dance music, with a spacey, otherworldly sound. Induces the experience of being lost in space, and the music would be designed to evoke a sense of wonder and awe, while being danceable.',
        'A rising synth is playing an arpeggio with a lot of reverb. It is backed by pads, sub bass line and soft drums. This song is full of synth sounds creating a soothing and adventurous atmosphere. It may be playing at a festival during two songs for a buildup.',
        'Slow tempo, bass-and-drums-led reggae song. Sustained electric guitar. High-pitched bongos with ringing tones. Vocals are relaxed with a laid-back feel, very expressive.',
    ],
    ['song with synths and flute', 'crowd cheering', 'piano sonata waltz, glittery', 'house song, 4 on the floor, rhythm'],
    ['chirping of birds and the distant echos of bells', 'cat meowing', 'saxophone with drums', 'beethoven piano sonata']
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference on trained musiclm model')

    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json', help='path to model config')
    parser.add_argument('--semantic_path', required=True, help='path to semantic stage checkpoint')
    parser.add_argument('--coarse_path', required=True, help='path to coarse stage checkpoint')
    parser.add_argument('--fine_path', required=True, help='path to fine stage checkpoint')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--return_coarse_wave', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--duration', default=4, type=float, help='duration of audio to generate in seconds')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)

    semantic_path = args.semantic_path
    coarse_path = args.coarse_path
    fine_path = args.fine_path
    return_coarse_wave = args.return_coarse_wave
    duration = args.duration
    kmeans_path = args.kmeans_path
    rvq_path = args.rvq_path
    seed = args.seed

    print(f'semantic checkpoint {semantic_path}, coarse checkpoint {coarse_path}, fine checkpoint {fine_path}')
    print(f'kmeans path {kmeans_path}, rvq path {rvq_path}, return_coarse_wave {return_coarse_wave}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    musiclm = create_musiclm_from_config(
        model_config=model_config,
        semantic_path=semantic_path,
        coarse_path=coarse_path,
        fine_path=fine_path,
        rvq_path=rvq_path,
        kmeans_path=kmeans_path,
        device=device)

    torch.manual_seed(seed)

    print('generating...')

    for prompt in prompts:
        generated_wave = musiclm.forward(
            text=prompt,
            output_seconds=duration,
            semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds,
            coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds,
            fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds,
            semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
            acoustic_steps_per_second=model_config.encodec_cfg.output_hz,
            return_coarse_generated_wave=return_coarse_wave,
        ).detach().cpu()

        print(generated_wave.shape)

        generated_wave = rearrange(generated_wave, 'b n -> b 1 n')
        for i, wave in enumerate(generated_wave):
            torchaudio.save(f'results/{prompt[i][:25]}_generated.wav', wave, musiclm.neural_codec.sample_rate)

