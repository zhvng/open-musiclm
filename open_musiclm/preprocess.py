import itertools
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from typing_extensions import Annotated
import time

from .clap_quantized import ClapQuantized
from .hf_hubert_kmeans import HfHubertWithKmeans, learn_kmeans
from .data import SoundDataset, get_dataloader
from .model_types import NeuralCodec, Wav2Vec
from .open_musiclm import (get_or_compute_clap_token_ids, get_or_compute_semantic_token_ids, get_or_compute_acoustic_token_ids)
from .optimizer import get_optimizer, get_linear_scheduler
from .utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, ceil_div, default,
                    eval_decorator, exists, generate_mask_with_prob,
                    get_embeds, gumbel_sample, mask_out_after_eos_id,
                    round_down_nearest_multiple, top_k)

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


@beartype
class DataPreprocessor(nn.Module):
    """
    Class to preprocess data for the single stage transformer trainer.
        Loads audio and saves the necessary tokens for the semantic, coarse and fine stages.
        Run this once for the target number of training samples (num_shards * batches_per_shard * batch_size) and use to train the model.
    """

    def __init__(
        self,
        *,
        num_shards,
        batch_size,
        batches_per_shard: int = 1,
        stage: Literal['all', 'semantic', 'coarse', 'fine']='all',
        num_coarse_quantizers=3,
        valid_frac=0.05,
        semantic_audio_length_seconds: float = 10.,
        coarse_audio_length_seconds: float = 4.,
        fine_audio_length_seconds: float = 2.,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors=True,
        folder=None,
        random_split_seed=42,
        results_folder='./data/fma_preprocessed',
        accelerate_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec
        self.num_coarse_quantizers = num_coarse_quantizers

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_shards = num_shards
        self.batch_size = batch_size
        self.accumulate_batches = batches_per_shard 
        self.valid_frac = valid_frac

        self.semantic_audio_length_seconds = semantic_audio_length_seconds
        self.coarse_audio_length_seconds = coarse_audio_length_seconds
        self.fine_audio_length_seconds = fine_audio_length_seconds

        # create dataset, same settings as trainer. TODO: process entire audio and crop later

        assert exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec)
        shards = [
            'semantic', 'coarse', 'fine'
        ]
        self.ds_fields_all = [
            ('raw_wave_for_clap', 'raw_wave_for_semantic'),
            ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic'),
            ('raw_wave_for_clap', 'raw_wave_for_acoustic'),
        ]
        target_sample_hz_all = [
            (audio_conditioner.sample_rate, wav2vec.target_sample_hz),
            (audio_conditioner.sample_rate, wav2vec.target_sample_hz, neural_codec.sample_rate),
            (audio_conditioner.sample_rate, neural_codec.sample_rate),
        ]
        normalize_all = [
            (False, True),
            (False, True, False),
            (False, False),
        ]
        seq_len_multiple_of = wav2vec.seq_len_multiple_of

        data_max_length_seconds_all = [
            (semantic_audio_length_seconds, semantic_audio_length_seconds),
            (semantic_audio_length_seconds, coarse_audio_length_seconds, coarse_audio_length_seconds),
            (semantic_audio_length_seconds, fine_audio_length_seconds)
        ]

        # narrow down to the stage we want to preprocess
        if stage != 'all':
            if stage == 'semantic':
                select_idx = 0
            elif stage == 'coarse':
                select_idx = 1
            elif stage == 'fine':
                select_idx = 2

            shards = shards[select_idx: select_idx + 1]
            self.ds_fields_all = self.ds_fields_all[select_idx: select_idx + 1]
            target_sample_hz_all = target_sample_hz_all[select_idx: select_idx + 1]
            normalize_all = normalize_all[select_idx: select_idx + 1]
            data_max_length_seconds_all = data_max_length_seconds_all[select_idx: select_idx + 1]

        assert exists(folder), 'audio folder must be passed in for preprocessing'

        self.ds = []
        for data_max_length_seconds, target_sample_hz, normalize in zip(
            data_max_length_seconds_all, target_sample_hz_all, normalize_all
        ):
            self.ds.append(SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                normalize=normalize,
                target_sample_hz=target_sample_hz,
                seq_len_multiple_of=seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            ))

        # split for validation

        self.valid_ds = []
        if valid_frac > 0:
            for i, ds in enumerate(self.ds):
                train_size = int((1 - valid_frac) * len(ds))
                valid_size = len(ds) - train_size
                ds, valid_ds = random_split(
                    ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
                self.ds[i] = ds
                self.valid_ds.append(valid_ds)
                self.print(
                    f'preprocessing a training dataset of {len(ds)} samples and validation set of randomly splitted {len(valid_ds)} samples')
        else:
            for ds in self.ds:
                self.valid_ds.append(ds)
                self.print(f'preprocessing a shared training and valid dataset of {len(ds)} samples')

        # dataloader

        self.dl = [get_dataloader(ds, batch_size=batch_size, shuffle=True) for ds in self.ds]

        self.valid_dl = [get_dataloader(self.valid_ds, batch_size=batch_size, shuffle=True) for valid_ds in self.valid_ds]

        # prepare
        (
            self.dl,
            self.valid_dl,
            self.audio_conditioner,
            self.neural_codec,
            self.wav2vec,
        ) = self.accelerator.prepare(
            self.dl,
            self.valid_dl,
            self.audio_conditioner,
            self.neural_codec,
            self.wav2vec,
        )

        # dataloader iterators

        self.dl_iter = [cycle(dl) for dl in self.dl]
        self.valid_dl_iter = [cycle(valid_dl) for valid_dl in self.valid_dl]

        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.shard_folders = [self.results_folder / s for s in shards]
        if self.is_main:
            for folder in self.shard_folders:
                if len([*folder.glob('**/*')]) > 0 and yes_or_no(f'existing directory found {folder}. do you want to clear it?'):
                    rmtree(str(folder))
                folder.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        self.shards = shards

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

    def generate_tokens_from_batch(self, raw_wave_for_clap=None, raw_wave_for_semantic=None, raw_wave_for_acoustic=None):
        device = self.device

        clap_token_ids = get_or_compute_clap_token_ids(None, self.accelerator.unwrap_model(self.audio_conditioner), raw_wave_for_clap.to(device), None) if exists(raw_wave_for_clap) else None
        semantic_token_ids = get_or_compute_semantic_token_ids(None, raw_wave_for_semantic.to(device), self.accelerator.unwrap_model(self.wav2vec)) if exists(raw_wave_for_semantic) else None
        coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(None, None, raw_wave_for_acoustic.to(device), self.accelerator.unwrap_model(self.neural_codec), self.num_coarse_quantizers) if exists(raw_wave_for_acoustic) else (None, None)

        return clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids)

    def generate_shard(self, accumulate_batches, ds_fields, dl_iter, shard_name):
        accumulate_batches = accumulate_batches // self.accelerator.num_processes # split up per process
        shard = None
        for _ in tqdm(range(accumulate_batches), desc=f'processing data for {shard_name}'):
            data_kwargs = dict(zip(ds_fields, next(dl_iter)))
            non_empty_batch = False
            while non_empty_batch is False:
                if len(data_kwargs) == 0:
                    continue

                clap_token_ids, semantic_token_ids, (coarse_token_ids, fine_token_ids) = self.generate_tokens_from_batch(**data_kwargs)

                if not exists(shard):
                    shard = {}
                
                if exists(clap_token_ids):
                    print(clap_token_ids.shape)
                    clap_token_ids = self.accelerator.gather_for_metrics(clap_token_ids.contiguous())
                    self.print(f'clap_token_ids.shape: {clap_token_ids.shape}')
                    clap_token_ids = clap_token_ids.detach().cpu().numpy()
                    shard['clap_token_ids'] = np.concatenate(without_none([shard.get('clap_token_ids'), clap_token_ids]), axis=0)

                if exists(semantic_token_ids):
                    semantic_token_ids = self.accelerator.gather_for_metrics(semantic_token_ids.contiguous())
                    semantic_token_ids = semantic_token_ids.detach().cpu().numpy()
                    shard['semantic_token_ids'] = np.concatenate(without_none([shard.get('semantic_token_ids'), semantic_token_ids]), axis=0)

                if exists(coarse_token_ids):
                    coarse_token_ids = self.accelerator.gather_for_metrics(coarse_token_ids.contiguous())
                    coarse_token_ids = coarse_token_ids.detach().cpu().numpy()
                    shard['coarse_token_ids'] = np.concatenate(without_none([shard.get('coarse_token_ids'), coarse_token_ids]), axis=0)
                
                if exists(fine_token_ids) and shard_name != 'coarse':
                    fine_token_ids = self.accelerator.gather_for_metrics(fine_token_ids.contiguous())
                    fine_token_ids = fine_token_ids.detach().cpu().numpy()
                    shard['fine_token_ids'] = np.concatenate(without_none([shard.get('fine_token_ids'), fine_token_ids]), axis=0)

                non_empty_batch = True

        return shard

    def processing_step(self):
        steps = int(self.steps.item())

        # logs

        logs = {}

        # collect and save data
        for ds_fields, dl_iter, shard_folder, shard_name in zip(self.ds_fields_all, self.dl_iter, self.shard_folders, self.shards):
            self.print(f'processing train shard and saving in {shard_folder}')
            train_shard = self.generate_shard(self.accumulate_batches, ds_fields, dl_iter, shard_name)
            self.print(f"saving train shard containing {self.accumulate_batches * self.batch_size} samples...")
            if self.is_main:
                np.save(shard_folder / f'train_{steps}.npy', train_shard)

        for ds_fields, valid_dl_iter, shard_folder, shard_name in zip(self.ds_fields_all, self.valid_dl_iter, self.shard_folders, self.shards):
            self.print(f'processing valid shard and saving in {shard_folder}')
            num_valid_batches = int(self.valid_frac * self.accumulate_batches)
            shard = self.generate_shard(num_valid_batches, ds_fields, valid_dl_iter, shard_name)
            self.print(f"saving valid shard containing {num_valid_batches * self.batch_size} samples...")
            if self.is_main:
                np.save(shard_folder / f'valid_{steps}.npy', shard)

        self.steps += 1

        return logs

    def process(self, log_fn=noop):

        while self.steps < self.num_shards:
            self.print(f"{int(self.steps.item())} out of {self.num_shards} shards processed")
            logs = self.processing_step()
            log_fn(logs)

        self.print('processing complete')
