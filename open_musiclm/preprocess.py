import io
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator, DistributedType
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from .clap_quantized import ClapQuantized
from .data import (SoundDatasetForPreprocessing,
                   get_sound_preprocessing_dataloader, init_sqlite)
from .hf_hubert_kmeans import HfHubertWithKmeans, learn_kmeans
from .model_types import NeuralCodec, Wav2Vec
from .open_musiclm import (get_or_compute_acoustic_token_ids,
                           get_or_compute_clap_token_ids,
                           get_or_compute_semantic_token_ids)
from .optimizer import get_linear_scheduler, get_optimizer
from .utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, beartype_jit, ceil_div,
                    copy_file_to_folder, default, eval_decorator, exists,
                    generate_mask_with_prob, get_embeds, gumbel_sample,
                    mask_out_after_eos_id, round_down_nearest_multiple, top_k)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


def noop(*args, **kwargs):
    pass

def without_none(arr):
    return list(filter(lambda x: x is not None, arr))

@beartype_jit
class DataPreprocessor(nn.Module):
    """
    Class to preprocess audio files for the single stage transformer trainer.

    Load audio and compute:
        1) clap tokens for the entire audio file, computed for 10 second sliding windows with 1 second interval
        2) semantic tokens for the entire audio file
        3) coarse+fine tokens for the entire audio file
    Run this once over the dataset and then use the preprocessed data for training.
    """

    def __init__(
        self,
        *,
        num_coarse_quantizers=3,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        max_audio_length_seconds=180,
        random_crop=True,
        clap_audio_length_seconds=10,
        semantic_audio_length_seconds=10,
        clap_batch_size=32,
        num_crops=1,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors=True,
        replace_existing=False,
        folder=None,
        results_folder='./data/fma_preprocessed',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec
        self.num_coarse_quantizers = num_coarse_quantizers
        self.max_audio_length_seconds = max_audio_length_seconds
        self.clap_audio_length_seconds = int(clap_audio_length_seconds)
        self.semantic_audio_length_seconds = int(semantic_audio_length_seconds)
        # TODO: allow a smaller clap length than semantic length, and average the clap embeddings over the time period as in the paper
        assert self.clap_audio_length_seconds == self.semantic_audio_length_seconds, 'clap window must be equal to semantic window for now'
        self.clap_batch_size = clap_batch_size
        self.num_crops = num_crops
        self.replace_existing = replace_existing

        self.register_buffer('steps', torch.Tensor([0]))

        # create dataset

        assert exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec)

        self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic')

        target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz, neural_codec.sample_rate)

        normalize = (False, True, False)

        seq_len_multiple_of = (None, wav2vec.seq_len_multiple_of, None)

        data_max_length_seconds = (max_audio_length_seconds, max_audio_length_seconds, max_audio_length_seconds)

        assert exists(folder), 'audio folder must be passed in for preprocessing'

        self.ds = SoundDatasetForPreprocessing(
            folder,
            pad_to_seconds=self.semantic_audio_length_seconds,
            max_length_seconds=data_max_length_seconds,
            random_crop=random_crop,
            normalize=normalize,
            target_sample_hz=target_sample_hz,
            seq_len_multiple_of=seq_len_multiple_of,
            ignore_load_errors=ignore_load_errors,
            ignore_files=ignore_files,
        )

        # dataloader

        self.dl = get_sound_preprocessing_dataloader(self.ds, batch_size=1, shuffle=False)

        # prepare

        (
            self.audio_conditioner,
            self.wav2vec,
            self.neural_codec,
            self.dl
        ) = self.accelerator.prepare(
            self.audio_conditioner,
            self.wav2vec,
            self.neural_codec,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.results_folder = Path(results_folder)

        if self.is_main:
            if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
                rmtree(str(self.results_folder))

            self.results_folder.mkdir(parents=True, exist_ok=True)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

        if self.is_main:
            self.conn, self.cursor = init_sqlite(str(self.results_folder / 'preprocessed.db'))
            self.cursor.execute("CREATE TABLE IF NOT EXISTS tokens(idx integer primary key, path text, clap array, semantic array, coarse array, fine array)")

        self.accelerator.wait_for_everyone()

        if not self.is_main:
            self.conn, self.cursor = init_sqlite(str(self.results_folder / 'preprocessed.db'))

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_tokens_from_batch(self, raw_wave_for_clap, raw_wave_for_semantic, raw_wave_for_acoustic):
        # split clap waveform into a clap_audio_length_seconds sliding window with 1 second interval. sample rate is self.audio_conditioner.sample_rate
        clap_split = raw_wave_for_clap.unfold(
            -1,
            self.audio_conditioner.sample_rate * self.clap_audio_length_seconds,
            self.audio_conditioner.sample_rate
        ).squeeze(0)

        batch_size = self.clap_batch_size
        clap_token_ids_all = []
        for i in range(0, clap_split.shape[0], batch_size):

            batch = clap_split[i:i+batch_size, :]
            clap_token_ids = get_or_compute_clap_token_ids(None, self.accelerator.unwrap_model(self.audio_conditioner), batch, None)
            clap_token_ids_all.append(clap_token_ids)

        clap_token_ids = torch.cat(clap_token_ids_all, dim=0)

        semantic_token_ids = get_or_compute_semantic_token_ids(None, raw_wave_for_semantic, self.accelerator.unwrap_model(self.wav2vec))

        coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(None, None, raw_wave_for_acoustic, self.accelerator.unwrap_model(self.neural_codec), self.num_coarse_quantizers)

        return clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids)

    def process(self, log_fn=noop):
        iters = math.ceil(self.num_crops * len(self.ds) / self.accelerator.num_processes)
        for idx in tqdm(range(iters), desc='processing data', mininterval=5):
            inputs = next(self.dl_iter)
            if exists(inputs):
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
                if not self.replace_existing:
                    self.cursor.execute("SELECT * FROM tokens WHERE idx=?", (idx,))
                    if len(self.cursor.fetchall()) > 0:
                        continue

                data_kwargs = dict(zip(self.ds_fields, inputs['data']))
                clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids) = self.generate_tokens_from_batch(**data_kwargs)

                clap_token_ids = clap_token_ids.detach().cpu().numpy()
                semantic_token_ids = semantic_token_ids.detach().cpu().numpy()
                coarse_token_ids = coarse_token_ids.detach().cpu().numpy()
                fine_token_ids = fine_token_ids.detach().cpu().numpy()

                # convert to int16 to save space
                clap_token_ids = clap_token_ids.astype(np.uint16)
                semantic_token_ids = semantic_token_ids.astype(np.uint16)
                coarse_token_ids = coarse_token_ids.astype(np.uint16)
                fine_token_ids = fine_token_ids.astype(np.uint16)
                # add tokens to sqlite db
                self.cursor.execute("INSERT INTO tokens VALUES (?, ?, ?, ?, ?, ?)", (idx, inputs['file_path'][0], clap_token_ids, semantic_token_ids, coarse_token_ids, fine_token_ids))
                self.conn.commit()

            self.steps += 1

        self.print('processing complete')
