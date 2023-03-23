from functools import partial, wraps
from pathlib import Path
from beartype.typing import Literal
from itertools import cycle

import torch
import numpy as np
import torch.nn.functional as F
import torchaudio
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union, List
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchaudio.functional import resample

from .utils import curtail_to_multiple, int16_to_float32, float32_to_int16, zero_mean_unit_var_norm, default

# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# type

OptionalIntOrTupleInt = Optional[Union[int, Tuple[Optional[int], ...]]]
FloatOrInt = Union[float, int]
# dataset functions

@beartype
class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3'],
        max_length_seconds: Optional[Union[FloatOrInt, Tuple[Optional[FloatOrInt], ...]]] = 1,
        normalize: Union[bool, Tuple[bool, ...]] = False,
        target_sample_hz: OptionalIntOrTupleInt = None,
        seq_len_multiple_of: OptionalIntOrTupleInt = None,
        ignore_files: Optional[List[str]] = None,
        ignore_load_errors=True,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = []
        ignore_files = default(ignore_files, [])
        for ext in exts:
            for file in path.glob(f'**/*.{ext}'):
                if any(ignore_file in str(file) for ignore_file in ignore_files):
                    print(f'found ignored file, skipping')
                    continue
                else:
                    files.append(file)
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.ignore_load_errors = ignore_load_errors

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length_seconds = cast_tuple(max_length_seconds, num_outputs)
        self.max_length = tuple([int(s * hz) if exists(s) else None for s, hz in zip(self.max_length_seconds, self.target_sample_hz)])

        self.normalize = cast_tuple(normalize, num_outputs)

        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(self.max_length_seconds) == len(
            self.target_sample_hz) == len(self.seq_len_multiple_of) == len(self.normalize)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file = self.files[idx]
            data, sample_hz = torchaudio.load(file)
        except:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                raise Exception(f'error loading file {file}')

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified 

        audio_length = data.size(1)
        temp_data = data
        temp_data_normalized = zero_mean_unit_var_norm(data)

        num_outputs = len(self.target_sample_hz)
        data = [None for _ in range(num_outputs)]

        sorted_max_length_seconds = sorted(
            enumerate(self.max_length_seconds), 
            key=lambda t: (t[1] is not None, t[1])) # sort by max_length_seconds, while moving None to the beginning

        for unsorted_i, max_length_seconds in sorted_max_length_seconds: 

            if exists(max_length_seconds):
                target_length = int(max_length_seconds * sample_hz)

                if audio_length > target_length:
                    max_start = audio_length - target_length
                    start = torch.randint(0, max_start, (1, ))

                    temp_data = temp_data[:, start:start + target_length]
                    temp_data_normalized = temp_data_normalized[:, start:start + target_length]
                else:
                    temp_data = F.pad(temp_data, (0, target_length - audio_length), 'constant')
                    temp_data_normalized = F.pad(temp_data_normalized, (0, target_length - audio_length), 'constant')

            data[unsorted_i] = temp_data_normalized if self.normalize[unsorted_i] else temp_data

        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))
        # quantize non-normalized audio to a valid waveform
        data_tuple = tuple(d if self.normalize[i] else int16_to_float32(float32_to_int16(d)) for i, d in enumerate(data_tuple))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            if exists(max_length):
                assert audio_length == max_length, f'audio length {audio_length} does not match max_length {max_length}.'

            data = rearrange(data, '1 ... -> ...')

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
        data = list(filter(lambda x: x is not None, data))
        if len(data) == 0:
            return () # empty batch

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

# dataloader to return data with mask (for variable length sequences)

@collate_one_or_multiple_tensors
def masked_pad_to_longest_fn(data):
    lengths = torch.tensor([t.shape[0] for t in data])
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)

    mask = torch.zeros_like(data, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    return {
        'input_values': data,
        'attention_mask': mask
    }

def get_masked_dataloader(ds, **kwargs):
    """
    pad to longest and return mask

    format: tuple({
        'input_values': torch.tensor,
        'attention_mask': torch.tensor
    })
    
    """
    collate_fn = masked_pad_to_longest_fn
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)


@beartype
class PreprocessedDataset(IterableDataset):
    def __init__(
        self,
        folder,
        stage: Literal['semantic', 'coarse', 'fine'],
        ds_fields: Tuple[str, ...],
        split: Literal['train', 'valid']
    ):
        super().__init__()
        path = Path(folder) / stage
        assert path.exists(), 'preprocessed data does not exist. run ./scripts/preprocess_data.py first.'

        shards = []
        for shard in path.glob(f'{split}_*.npy'):
            shards.append(shard)
        assert len(shards) > 0, 'no shards found'
        shards = sorted(shards)

        self.shards = cycle(shards)
        self.ds_fields = ds_fields

    def load_shard(self, file_path):
        return np.load(file_path, allow_pickle=True)

    def __iter__(self):
        while exists(x := next(self.shards, None)):
            loaded_shard = self.load_shard(x).item()

            if not exists(loaded_shard):
                raise ValueError(f'found empty shard {x}. it is likely that batches_per_shard was too small when preprocessing the data.')

            data = []
            data_length = None
            for key in self.ds_fields:
                val = loaded_shard[key]
                data.append(val)

                if exists(data_length):
                    assert data_length == len(val), f'data length mismatch in shard {x}'
                else:
                    data_length = len(val)

            for i in range(data_length):
                yield tuple(torch.tensor(val[i]) for val in data)

def get_preprocessed_dataloader(ds, **kwargs):
    return DataLoader(ds, **kwargs)