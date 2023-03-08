import os
import sys

import torchaudio
from torchaudio.functional import resample
import numpy as np
import torch
from transformers import RobertaTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from open_musiclm.clap_quantized import ClapQuantized, create_clap_quantized

text_data = ['male rap',
            'male rapping over a synth pad',
            'female singing',
            'female singing over a synth pad',
            'male singing over a synth pad',
            'male rapping chill voice',
            'male with deep voice',
            'male singing then rapping, synth in the background',
            'producer tag then drake rapping',
            'calming melody',
            'upbeat, hype',
            'pause and then the beat drops, and a male rapper is rapping over a trap beat',
            'male rapping over a hip hop beat',
            'house music',
            'rock song with piano',
            'groovy melody with piano and a male singing']

def infer_text(clap_wrapper, return_embedding=False):



    text_embed = clap_wrapper(text_input=text_data, return_embedding=return_embedding)

    return text_embed

def infer_audio(clap_wrapper: ClapQuantized, return_embedding: bool = False, device: str = 'cuda'):

    print('inferring audio...')

    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = torchaudio.load('/u/zhvng/projects/audio_files/jumpman.mp3')

    wave_2, sr_2 = torchaudio.load('/u/zhvng/projects/open-musiclm/data/fma_large/000/000048.mp3')

    if audio_waveform.shape[0] > 1:
        # the audio has more than 1 channel, convert to mono
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    if wave_2.shape[0] > 1:
        wave_2 = torch.mean(wave_2, dim=0, keepdim=True)

    audio_waveform = resample(audio_waveform, sr, 48000)
    wave_2 = resample(wave_2, sr_2, 48000)

    # audio_waveform = audio_waveform[:, :48000 * 30]
    audio_waveform_1 = audio_waveform[:, :48000 * 10]
    audio_waveform_2 = wave_2[:, :48000 * 10]
    # audio_waveform_3 = audio_waveform[:, 48000 * 20 : 48000 * 50]
    audio_waveform = torch.cat([audio_waveform_1, audio_waveform_2], dim=0)

    audio_embed = clap_wrapper(audio_input=audio_waveform.to(device), return_embedding=return_embedding)

    return audio_embed


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clap_wrapper = create_clap_quantized(device=device, learn_rvq=False).to(device)

    text_embeds = infer_text(clap_wrapper, return_embedding=True)
    audio_embed = infer_audio(clap_wrapper, return_embedding=True, device=device)

    # print(text_embeds)
    print(text_embeds.shape)

    # print(audio_embed)
    print(audio_embed.shape)

    for i, text_embed in enumerate(text_embeds):
        # get cosine similarity with audio_embed
        cos_sim = torch.nn.functional.cosine_similarity(
            audio_embed, text_embed, dim=-1)
        print(text_data[i], cos_sim.cpu().numpy())
