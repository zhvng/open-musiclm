from functools import partial, wraps
from pathlib import Path
from beartype.typing import Literal
from itertools import cycle
import sqlite3
import io
import random

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

# sqlite helpers for preprocessing

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

@beartype
def init_sqlite(db_path):
    """Connect to a sqlite database. Will create a new one if it doesn't exist."""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    return conn, cursor

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
            
        return self.process_audio(data, sample_hz)

    def process_audio(self, data, sample_hz):

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified 

        temp_data = data
        temp_data_normalized = zero_mean_unit_var_norm(data)

        num_outputs = len(self.target_sample_hz)
        data = [None for _ in range(num_outputs)]

        sorted_max_length_seconds = sorted(
            enumerate(self.max_length_seconds), 
            key=lambda t: (t[1] is not None, t[1])) # sort by max_length_seconds, while moving None to the beginning

        for unsorted_i, max_length_seconds in sorted_max_length_seconds: 

            if exists(max_length_seconds):
                audio_length = temp_data.size(1)
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


@beartype
class SoundDatasetForPreprocessing(SoundDataset):
    def __init__(
        self,
        folder,
        **kwargs
    ):
        super().__init__(folder=folder, **kwargs)

    def __getitem__(self, idx):
        """
        override getitem to return data + audio file name, and None in the case of a load error.
        intended to work with batch_size = 1 (for now). will likely be i/o limited anyways.
        """
        try:
            file = self.files[idx]
            data, sample_hz = torchaudio.load(file)
        except:
            if self.ignore_load_errors:
                return None
            else:
                raise Exception(f'error loading file {file}')

        # if audio length is less than 10 seconds, pad to 10 seconds
        # else pad audio to nearest second
        # this is so at least one semantic token sequence can be extracted from the audio
        # TODO: support longer context
        if data.size(1) < 10 * sample_hz:
            data = F.pad(data, (0, 10 * sample_hz - data.size(1)), 'constant', value=0)
        else:
            data = F.pad(data, (0, sample_hz - data.size(1) % sample_hz), 'constant', value=0)

        data = self.process_audio(data, sample_hz)

        return {
            'idx': idx,
            'data': data,
            'file_path': str(file)
        }

def sound_preprocessing_collate_fn(data):
    data = list(filter(lambda x: x is not None, data))
    if len(data) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(data)

def get_sound_preprocessing_dataloader(ds, **kwargs):
    assert kwargs.get('batch_size', 1) == 1, 'batch_size must be 1 for preprocessing'
    kwargs.setdefault('batch_size', 1)
    return DataLoader(ds, collate_fn=sound_preprocessing_collate_fn, **kwargs)

@beartype
class PreprocessedDataset(Dataset):
    def __init__(
        self,
        folder,
        stage: Literal['semantic', 'coarse', 'fine'],
        semantic_window_seconds: int=10,
        coarse_window_seconds: int=4,
        fine_window_seconds: int=2,
        semantic_steps_per_second=50,
        acoustic_steps_per_second=75,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.stage = stage
        self.semantic_window_seconds = semantic_window_seconds
        self.coarse_window_seconds = coarse_window_seconds
        self.fine_window_seconds = fine_window_seconds
        self.semantic_steps_per_second = semantic_steps_per_second
        self.acoustic_steps_per_second = acoustic_steps_per_second

        self.conn, self.cursor = init_sqlite(str(path / 'preprocessed.db'))
        self.cursor.execute('SELECT idx from tokens')
        self.ids = [i[0] for i in self.cursor.fetchall()]

    def __len__(self):
        return len(self.ids)

    def get_and_assert_audio_length_from_tokens(self, clap_token_ids=None, semantic_token_ids=None, coarse_token_ids=None, fine_token_ids=None):
        """compute original audio length from tokens and assert that all provided tokens have the same audio length"""
        clap_audio_length = clap_token_ids.shape[0] + 10 - 1 if exists(clap_token_ids) else None
        semantic_audio_length = (semantic_token_ids.shape[1] + 1) // self.semantic_steps_per_second if exists(semantic_token_ids) else None
        coarse_audio_length = coarse_token_ids.shape[1] // self.acoustic_steps_per_second if exists(coarse_token_ids) else None
        fine_audio_length = fine_token_ids.shape[1] // self.acoustic_steps_per_second if exists(fine_token_ids) else None

        lengths = [clap_audio_length, semantic_audio_length, coarse_audio_length, fine_audio_length]
        lengths = [int(l) for l in lengths if exists(l)]
        assert len(set(lengths)) == 1, 'audio lengths are not equal'

        return lengths[0]
    
    def get_clap_tokens(self, clap_token_ids, start_idx):
        """aggregate clap token over entire context with a sliding"""
        return clap_token_ids[start_idx].unsqueeze(0)

    def crop_semantic_tokens(self, semantic_token_ids, start_idx, end_idx):
        # with start_idx = 0, end_idx = 2, semantic_steps_per_second=50
        # we return semantic_token_ids[:, 0:99]
        return semantic_token_ids[:, start_idx * self.semantic_steps_per_second: end_idx * self.semantic_steps_per_second - 1]

    def crop_acoustic_tokens(self, coarse_or_fine_ids, start_idx, end_idx):
        # with start_idx = 0, end_idx = 2, coarse_steps_per_second=75
        # we return coarse_token_ids[:, 0:150]
        return coarse_or_fine_ids[:, start_idx * self.acoustic_steps_per_second: end_idx * self.acoustic_steps_per_second]

    def compute_crop_indices(self, audio_length, outside_window, inside_window=None):
        outside_start_idx = random.randint(0, audio_length - outside_window)
        outside_end_idx = outside_start_idx + outside_window
        
        if exists(inside_window):
            inside_start_idx = random.randint(outside_start_idx, outside_end_idx - inside_window)
            inside_end_idx = inside_start_idx + inside_window

            return outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx
        else:
            return outside_start_idx, outside_end_idx, None, None

    def __getitem__(self, idx):
        sqlite_idx = self.ids[idx]

        # load and crop the tokens
        # 1) select a outer crop (in whole seconds) from clap tokens based on semantic window, taking the average if spans multiple windows
        # 2) select an inner crop inside this range
        # 3) return cropped tokens

        # c - -
        #   c - -
        #     c - -
        #       c - -
        #         c - -
        #           c - -    <- clap tokens with window size 3 computed in a sliding window
        # s s s s s s s s    <- semantic tokens where 's' represents self.semantic_steps_per_second tokens
        # c c c c c c c c    <- coarse tokens where 'c' represents self.acoustic_steps_per_second tokens
        #           [    ]  <- 1st crop: start_idx=5, end_idx=8
        #           [   ]    <- 2nd crop: start_idx=5, end_idx=7
        # 0 1 2 3 4 5 6 7

        if self.stage == 'semantic':
            clap_token_ids, semantic_token_ids = self.cursor.execute(f'SELECT clap, semantic FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, semantic_token_ids = torch.from_numpy(clap_token_ids), torch.from_numpy(semantic_token_ids)

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, semantic_token_ids=semantic_token_ids) 

            outside_start_idx, outside_end_idx, _, _ = self.compute_crop_indices(audio_length, self.semantic_window_seconds) 

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx)
            semantic_token_ids = self.crop_semantic_tokens(semantic_token_ids, outside_start_idx, outside_end_idx)

            return (clap_token_ids, semantic_token_ids)
        elif self.stage == 'coarse':
            clap_token_ids, semantic_token_ids, coarse_token_ids = self.cursor.execute(f'SELECT clap, semantic, coarse FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, semantic_token_ids, coarse_token_ids = torch.from_numpy(clap_token_ids), torch.from_numpy(semantic_token_ids), torch.from_numpy(coarse_token_ids)

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, semantic_token_ids=semantic_token_ids, coarse_token_ids=coarse_token_ids) 

            outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(audio_length, self.semantic_window_seconds, self.coarse_window_seconds)

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx) 
            semantic_token_ids = self.crop_semantic_tokens(semantic_token_ids, inside_start_idx, inside_end_idx)
            coarse_token_ids = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)

            return (clap_token_ids, semantic_token_ids, coarse_token_ids)
        elif self.stage == 'fine':
            clap_token_ids, coarse_token_ids, fine_token_ids = self.cursor.execute(f'SELECT clap, coarse, fine FROM tokens where idx = ?', (sqlite_idx,)).fetchone()
            clap_token_ids, coarse_token_ids, fine_token_ids = torch.from_numpy(clap_token_ids), torch.from_numpy(coarse_token_ids), torch.from_numpy(fine_token_ids)

            audio_length = self.get_and_assert_audio_length_from_tokens(clap_token_ids=clap_token_ids, coarse_token_ids=coarse_token_ids, fine_token_ids=fine_token_ids) 

            outside_start_idx, outside_end_idx, inside_start_idx, inside_end_idx = self.compute_crop_indices(audio_length, self.semantic_window_seconds, self.fine_window_seconds)

            clap_token_ids = self.get_clap_tokens(clap_token_ids, outside_start_idx)
            coarse_token_ids = self.crop_acoustic_tokens(coarse_token_ids, inside_start_idx, inside_end_idx)
            fine_token_ids = self.crop_acoustic_tokens(fine_token_ids, inside_start_idx, inside_end_idx)

            return (clap_token_ids, coarse_token_ids, fine_token_ids)
        else:
            raise Exception(f'invalid stage {self.stage}')

@collate_one_or_multiple_tensors
def concatenate_fn(batch):
    return torch.cat(batch, dim=0)

def get_preprocessed_dataloader(ds, **kwargs):
    collate_fn = concatenate_fn
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)