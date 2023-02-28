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
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
import argparse
from dataclasses import asdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_coarse_transformer, create_fine_transformer, create_semantic_transformer, MusicLM
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.hf_hubert_kmeans import get_hubert_kmeans
from open_musiclm.config import load_model_config
from scripts.train_utils import disable_print

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

    print('loading clap...')
    with disable_print():
        clap = create_clap_quantized(
            device=device, 
            learn_rvq=False, 
            rvq_checkpoint_path=rvq_path,
            **asdict(model_config.clap_rvq_cfg),
        ).to(device)

    print('loading wav2vec...')
    wav2vec = get_hubert_kmeans(
        kmeans_path=kmeans_path,
        **asdict(model_config.hubert_kmeans_cfg),
    ).to(device)

    print('loading encodec')
    encodec_wrapper = create_encodec_24khz(**asdict(model_config.encodec_cfg)).to(device)

    print('creating transformers')
    semantic_transformer = create_semantic_transformer(
        clap_codebook_size=clap.codebook_size,
        semantic_codebook_size=wav2vec.codebook_size,
        **asdict(model_config.semantic_cfg),
    ).to(device)

    coarse_transformer = create_coarse_transformer(
        clap_codebook_size=clap.codebook_size,
        semantic_codebook_size=wav2vec.codebook_size,
        acoustic_codebook_size=encodec_wrapper.codebook_size,
        **asdict(model_config.coarse_cfg),
    ).to(device)

    fine_transformer = create_fine_transformer(
        clap_codebook_size=clap.codebook_size,
        acoustic_codebook_size=encodec_wrapper.codebook_size,
        **asdict(model_config.fine_cfg),
    ).to(device)

    def load_model(model, path):
        path = Path(path)
        assert path.exists(), f'checkpoint does not exist at {str(path)}'
        pkg = torch.load(str(path))
        model.load_state_dict(pkg)

    print('loading semantic stage...')
    load_model(semantic_transformer, semantic_path)

    print('loading coarse stage...')
    load_model(coarse_transformer, coarse_path)

    print('loading fine stage...')
    load_model(fine_transformer, fine_path)

    print('loading musiclm')
    musiclm = MusicLM(
        wav2vec=wav2vec,
        clap=clap,
        neural_codec=encodec_wrapper,
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        fine_transformer=fine_transformer,
    )

    torch.manual_seed(seed)

    print('generating...')

    for prompt in prompts:
        generated_wave = musiclm.forward(
            text=prompt, 
            output_seconds=duration,
            semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds, 
            coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds, 
            fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds, 
            return_coarse_generated_wave=return_coarse_wave,
        )

        print(generated_wave.shape)

        generated_wave = rearrange(generated_wave, 'b n -> b 1 n')
        for i, wave in enumerate(generated_wave):
            torchaudio.save(f'results/{prompt[i][:25]}_generated.wav', wave, encodec_wrapper.sample_rate)

