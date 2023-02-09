import os
import sys

import torchaudio
from torchaudio.functional import resample
import numpy as np
import torch
from transformers import RobertaTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from open_musiclm.clap import create_model
from open_musiclm.clap_quantized import ClapQuantized

# tokenize = RobertaTokenizer.from_pretrained('roberta-base')

# def tokenizer(text):
#     result = tokenize(
#         text,
#         padding="max_length",
#         truncation=True,
#         max_length=77,
#         return_tensors="pt",
#     )
#     return {k: v.squeeze(0) for k, v in result.items()}


def infer_text(clap_wrapper, return_embedding=False):

    # load the text, can be a list (i.e. batch size)
    # text_data = ["air horn",
    #              "high pitched air horn",
    #              "buzzing noise",
    #              "air horn with a buzzing noise",
    #              "air horn with a low pitched rumble",
    #              "horn"]

    text_data = ['male rap',
                 'male rapping over a synth pad',
                 'female singing',
                 'female singing over a synth pad',
                 'male rapping chill voice',
                 'male with deep voice',
                 'male singing then rapping, synth in the background',
                 'producer tag then drake rapping',
                 'drake',
                 'future',
                 'metro boomin']
    # tokenize for roberta, if you want to tokenize for another text encoder, please refer to data.py#L43-90

    text_embed = clap_wrapper(text_input=text_data, return_embedding=return_embedding)

    return text_embed


def int16_to_float32(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


def infer_audio(clap_wrapper: ClapQuantized, return_embedding: bool = False):

    print('inferring audio...')

    # load the waveform of the shape (T,), should resample to 48000
    audio_waveform, sr = torchaudio.load('/u/zhvng/projects/audio_files/jumpman.mp3')

    if audio_waveform.shape[0] > 1:
        # the audio has more than 1 channel, convert to mono
        audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

    # print(audio_waveform.shape, sr)
    audio_waveform = resample(audio_waveform, sr, 48000)
    # audio_waveform = audio_waveform.squeeze(0)
    # quantize
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    # audio_waveform = torch.from_numpy(audio_waveform).float()

    # audio_dict = {}

    # # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
    # audio_dict = get_audio_features(
    #     audio_dict, audio_waveform, 480000,
    #     data_truncating='fusion',
    #     data_filling='repeatpad',
    #     audio_cfg=clap_wrapper.clap_cfg['audio_cfg']
    # )

    audio_embed = clap_wrapper(audio_input=audio_waveform, return_embedding=return_embedding)

    return audio_embed


if __name__ == "__main__":

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)
    precision = 'fp32'
    amodel = 'HTSAT-tiny'  # or 'PANN-14'
    tmodel = 'roberta'  # the best text encoder in our training
    enable_fusion = True  # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    # the checkpoint name, the unfusion model can also be loaded.
    pretrained = "/u/zhvng/projects/clap-chkpt/laion_audioset_fusion/checkpoints/epoch_top_0.pt"

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )

    clap_wrapper = ClapQuantized(clap=model, clap_cfg=model_cfg)
    clap_wrapper = clap_wrapper.to(device)

    text_embeds = infer_text(clap_wrapper, return_embedding=True)
    audio_embed = infer_audio(clap_wrapper, return_embedding=True)

    print(text_embeds)
    print(text_embeds.size())

    print(audio_embed)
    print(audio_embed.size())

    for text_embed in text_embeds:
        # get cosine similarity with audio_embed
        cos_sim = torch.nn.functional.cosine_similarity(
            audio_embed, text_embed, dim=-1)
        print(cos_sim)
