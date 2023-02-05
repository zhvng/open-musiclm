from functools import partial, wraps
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from utils import curtail_to_multiple 


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


# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]

# dataset functions

@beartype
class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav'],
        max_length: OptionalIntOrTupleInt = None,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = [file for ext in exts for file in path.glob(f'**/*.{ext}')]
        assert len(files) > 0, 'no sound files found'

        self.files = files

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length = cast_tuple(max_length, num_outputs)
        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.target_sample_hz) == len(self.seq_len_multiple_of)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        num_outputs = len(self.target_sample_hz)
        data = cast_tuple(data, num_outputs)

        # resample if target_sample_hz is not None in the tuple

        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            # pad or curtail

            if audio_length > max_length:
                max_start = audio_length - max_length
                start = torch.randint(0, max_start, (1, ))
                data = data[:, start:start + max_length]

            else:
                data = F.pad(data, (0, max_length - audio_length), 'constant')

            data = rearrange(data, '1 ... -> ...')

            if exists(max_length):
                data = data[:max_length]

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]

        return output

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
