import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union
from clap import CLAP
from einops import rearrange
from torch import nn
from transformers import RobertaTokenizer
from utils import exists
from vector_quantize_pytorch import ResidualVQ


def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    )(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=64,
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    """
    with torch.no_grad():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                # the +1 related to how the spectrogram is computed
                chunk_frames = max_len // audio_cfg['hop_size']+1
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(
                        list(range(0, total_frames-chunk_frames+1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front+chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle +
                                           chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back+chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(
                        size=[chunk_frames, 64])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack(
                        [mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len/len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len/len(audio_data))
                    audio_data = audio_data.repeat(n_repeat+1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample


tokenize = RobertaTokenizer.from_pretrained('roberta-base')


def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

@beartype
class ClapQuantized(nn.Module):
    def __init__(self,
                 *,
                 clap: CLAP,
                 clap_cfg: Dict[str, any],
                 codebook_size: int = 1024,
                 rq_num_quantizers: int = 12,
                 rq_ema_decay: float = 0.95,
                 ):
        super().__init__()

        self.clap = clap
        self.clap_cfg = clap_cfg

        for param in self.clap.parameters():
            param.requires_grad = False

        self.rq = ResidualVQ(
            dim=clap.joint_embed_shape,
            num_quantizers=rq_num_quantizers,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
            commitment_weight=0,  # embeddings are frozen so no need for commitment loss
            decay=rq_ema_decay,
            kmeans_init=True,
            threshold_ema_dead_code=2,
        )

    def forward(self,
                *,
                audio_input: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
                text_input: Optional[List[str]] = None,
                ):
        """
        Wrapper for clap module that takes in audio or text and returns the quantized embedding from the respective tower
        """

        assert exists(audio_input) ^ exists(text_input), "either audio or text must be provided, but not both"

        with torch.no_grad():
            self.clap.eval()
            if exists(audio_input):
                audio_dicts = []
                for waveform in audio_input:
                    audio_dict = get_audio_features({}, waveform, 480000, data_truncating='fusion',
                                                    data_filling='repeatpad', audio_cfg=self.clap_cfg['audio_cfg'])
                    audio_dicts.append(audio_dict)

                embedding = self.clap.get_audio_embedding(audio_dicts)
            else:
                text_input = tokenizer(text_input)
                embedding = self.clap.get_text_embedding(text_input)

        print(embedding.shape)

        _, indices, _ = self.rq(rearrange(embedding, 'n c -> 1 n c'))

        return indices
