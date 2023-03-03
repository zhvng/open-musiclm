import itertools
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
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
from .open_musiclm import (CoarseStage, FineStage, SemanticStage,
                           TokenConditionedTransformer)
from .optimizer import get_optimizer, get_linear_scheduler
from .utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, ceil_div, default,
                    eval_decorator, exists, generate_mask_with_prob,
                    get_embeds, gumbel_sample, mask_out_after_eos_id,
                    round_down_nearest_multiple, top_k)

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    input_audio=Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


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


@beartype
class SingleStageTrainer(nn.Module):
    """
    General trainer for any stage of MusicLM.
        semantic: requires audio_conditioner and wav2vec
        coarse: requires audio_conditioner, wav2vec, and neural_codec
        fine: requires audio_conditioner and neural_codec
    """

    def __init__(
        self,
        transformer: TokenConditionedTransformer,
        stage: Literal['semantic', 'coarse', 'fine'],
        *,
        num_train_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        data_max_length_seconds = 1,
        ignore_files: Optional[List[str]]=None,
        cross_entropy_loss_weights: Optional[List[float]]=None,
        ignore_load_errors=True,
        folder=None,
        lr=3e-4,
        lr_warmup=0,
        grad_accum_every=1,
        wd=0.,
        max_grad_norm=0.5,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_predicted_tokens=True,
        save_reconstructed_wave=True,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = {}
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.transformer = transformer
        self.wav2vec = wav2vec
        self.transformer = transformer
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec

        self.stage = stage

        if stage == 'semantic':
            assert exists(audio_conditioner) and exists(wav2vec)
            self.train_wrapper = SemanticStage(
                semantic_transformer=transformer,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 1.])
            )
            self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic')
            target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz)
            normalize = (False, True)
            seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'coarse':
            assert exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec)
            self.train_wrapper = CoarseStage(
                coarse_transformer=transformer,
                neural_codec=neural_codec,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic')
            target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz, neural_codec.sample_rate)
            normalize = (False, True, False)
            seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'fine':
            assert exists(audio_conditioner) and exists(neural_codec)
            self.train_wrapper = FineStage(
                fine_transformer=transformer,
                clap=audio_conditioner,
                neural_codec=neural_codec,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_acoustic')
            target_sample_hz = (audio_conditioner.sample_rate, neural_codec.sample_rate)
            normalize = (False, False)
            seq_len_multiple_of = None
        else:
            raise ValueError(f'invalid stage: {stage}')

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr=lr, wd=wd)

        if lr_warmup > 0:
            self.scheduler = get_linear_scheduler(
                self.optim,
                total_iters=lr_warmup,
            )
        else:
            self.scheduler = None

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset
        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                normalize=normalize,
                target_sample_hz=target_sample_hz,
                seq_len_multiple_of=seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=batch_size, shuffle=True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size=batch_size, shuffle=True)

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.save_predicted_tokens = save_predicted_tokens
        self.save_reconstructed_wave = save_reconstructed_wave

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        if exists(save_reconstructed_wave):
            self.waves_folder = self.results_folder / 'reconstructed_waves'
            self.waves_folder.mkdir(parents=True, exist_ok=True)
        if exists(save_predicted_tokens):
            self.tokens_folder = self.results_folder / 'tokens'
            self.tokens_folder.mkdir(parents=True, exist_ok=True)

        hps = {"num_train_steps": num_train_steps, "learning_rate": lr}
        self.accelerator.init_trackers(f"{stage}_stage_{int(time.time() * 1000)}", config=hps)

    def save(self, model_path, optim_path, scheduler_path=None):
        model_state_dict = self.accelerator.get_state_dict(self.transformer)
        torch.save(model_state_dict, model_path)

        optim_state_dict = self.optim.state_dict()
        torch.save(optim_state_dict, optim_path)

        if exists(self.scheduler):
            assert exists(scheduler_path)
            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict, scheduler_path)
      
    def load(self, model_path, optim_path, scheduler_path=None):
        model_path = Path(model_path)
        optim_path = Path(optim_path)
        assert model_path.exists() and optim_path.exists()

        model_state_dict = torch.load(model_path, map_location=self.device)
        optim_state_dict = torch.load(optim_path, map_location=self.device)
        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(model_state_dict)
        self.optim.load_state_dict(optim_state_dict)

        if exists(self.scheduler):
            assert exists(scheduler_path), 'the config specifies lr warmup is used, but no scheduler checkpoint is given. try setting lr_warmup to 0.'
            scheduler_path = Path(scheduler_path)
            assert scheduler_path.exists()
            scheduler_state_dict = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state_dict)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

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

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))
            non_empty_batch = False
            while non_empty_batch is False:
                if len(data_kwargs) == 0:
                    continue

                loss, _, _ = self.train_wrapper(**data_kwargs, return_loss=True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

                non_empty_batch = True

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        if exists(self.scheduler):
            self.scheduler.step()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        if self.is_main and not (steps % self.save_results_every):
            non_empty_batch = False
            while non_empty_batch is False:
                data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))
                if len(data_kwargs) == 0:
                    continue
                
                with torch.no_grad():
                    self.train_wrapper.eval()
                    valid_loss, all_logits, all_labels = self.train_wrapper(**data_kwargs, return_loss=True)

                self.print(f'{steps}: valid loss {valid_loss}')
                self.accelerator.log({"valid_loss": valid_loss}, step=steps)

                if self.save_predicted_tokens:
                    # interleave pred_tokens and gt_tokens and save to a text file

                    assert exists(self.tokens_folder)

                    pred_tokens = all_logits[-1].detach().cpu().argmax(1).long()
                    gt_tokens = all_labels[-1].detach().cpu().long()

                    interleave = torch.empty((pred_tokens.shape[0] + gt_tokens.shape[0], pred_tokens.shape[1]), dtype=pred_tokens.dtype)
                    interleave[0::2] = pred_tokens
                    interleave[1::2] = gt_tokens

                    np.savetxt(str(self.tokens_folder / f'{self.stage}.tokens.{steps}.txt'), interleave, fmt='%-6s', header='predicted and ground truth tokens from the validation set. row 0%2 is predicted, 1%2 is ground truth\n ')

                    non_empty_batch = True

                if self.save_reconstructed_wave and (self.stage == 'coarse' or self.stage=='fine'):
                    # For coarse and fine stages, reconstruct teacher-forced wave from logits 

                    assert exists(self.neural_codec)
                    assert exists(self.waves_folder)

                    pred_tokens = all_logits[-1].detach().argmax(1)[:, :-1]
                    pred_tokens[pred_tokens == self.transformer.eos_ids[-1]] = 0

                    num_quantizers = self.transformer.token_sequences[-1].num_quantizers
                    pred_tokens = rearrange(pred_tokens, 'b (n q) -> b n q', q=num_quantizers)

                    if self.stage == 'fine':
                        coarse_tokens = all_labels[-2][:, :-1]
                        coarse_quantizers = self.transformer.token_sequences[-2].num_quantizers
                        coarse_tokens = rearrange(coarse_tokens, 'b (n q) -> b n q', q=coarse_quantizers)
                        pred_tokens = torch.cat((coarse_tokens, pred_tokens), dim=-1)
                    
                    waves = self.neural_codec.decode_from_codebook_indices(pred_tokens)
                    waves = waves.cpu()
                    for i, wave in enumerate(waves):
                        torchaudio.save(str(self.waves_folder / f'{self.stage}.reconstructed_wave_{i}.{steps}.wav'), wave, self.neural_codec.sample_rate)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

            model_path = str(self.results_folder / f'{self.stage}.transformer.{steps}.pt')
            optim_path = str(self.results_folder / f'{self.stage}.optimizer.{steps}.pt')
            scheduler_path = str(self.results_folder / f'{self.stage}.scheduler.{steps}.pt')

            self.save(model_path, optim_path, scheduler_path)

            # save audio conditioner (clap) rvq checkpoint
            if self.audio_conditioner.learn_rvq:
                rvq_state_dict = self.audio_conditioner.rq.state_dict()
                torch.save(rvq_state_dict, str(self.results_folder / f'{self.stage}.conditioner_rvq.{steps}.pt'))
                
        self.steps += 1
        return logs

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')


@beartype
class ClapRVQTrainer(nn.Module):
    """
    Learn the residual vector quantizer to turn CLAP embeddings into discrete tokens.
    """

    def __init__(
        self,
        *,
        num_train_steps,
        batch_size,
        accumulate_batches: Optional[int] = None, 
        audio_conditioner: Optional[ClapQuantized] = None,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        wd=0.,
        max_grad_norm=0.5,
        data_max_length_seconds: Union[float, int] = 10,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_model_every=1000,
        results_folder='./results',
    ):
        super().__init__()

        self.audio_conditioner = audio_conditioner
        self.ds = dataset
        self.num_train_steps = num_train_steps
        self.accumulate_batches = accumulate_batches
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                target_sample_hz=audio_conditioner.sample_rate,
                seq_len_multiple_of=None,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=batch_size, shuffle=True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size=batch_size, shuffle=True)

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def print(self, msg):
        print(msg)

    @property
    def device(self):
        return next(self.parameters()).device

    def train_step(self):
        steps = int(self.steps.item())

        self.audio_conditioner.learn_rvq = True
    
        iters = default(self.accumulate_batches, 1)

        embeds = []
        for _ in range(iters):
            raw_wave_for_clap = next(self.dl_iter)[0]
            embed = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_embedding=True)
            embeds.append(embed)
        
        embeds = torch.cat(embeds, dim=0)

        loss = self.audio_conditioner.quantize(embeds, return_rvq_loss=True)
        self.print(f'loss: {loss}')

        # sample results every so often
        if not (steps % self.save_results_every):
            raw_wave_for_clap = next(self.valid_dl_iter)[0]

            with torch.no_grad():
                self.audio_conditioner.learn_rvq = False
                valid_loss = self.audio_conditioner.forward(audio_input=raw_wave_for_clap.to(self.device), return_rvq_loss=True)

            self.print(f'{steps}: valid loss {valid_loss}')

        # save model every so often

        if not (steps % self.save_model_every):
            # save audio conditioner (clap) rvq checkpoint
            rvq_state_dict = self.audio_conditioner.rq.state_dict()
            torch.save(rvq_state_dict, str(self.results_folder / f'clap.rvq.{steps}.pt'))

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')


@beartype
class HfHubertKmeansTrainer(nn.Module):
    """
    Trainer for kmeans part of HfHubertWithKmeans. Consists of two parts: 1) extracting Hubert features and 2) training kmeans model on these features.
    """

    def __init__(
        self,
        *,
        feature_extraction_num_steps: int,
        feature_extraction_batch_size: int,
        hubert_kmeans: HfHubertWithKmeans,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        data_max_length_seconds: Union[float, int] = 1,
        results_folder='./results',
    ):
        super().__init__()

        self.ds = dataset
        self.feature_extraction_num_steps = feature_extraction_num_steps
        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.hubert_kmeans = hubert_kmeans
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            self.ds = SoundDataset(
                folder,
                max_length_seconds=data_max_length_seconds,
                normalize=True,
                target_sample_hz=hubert_kmeans.target_sample_hz,
                seq_len_multiple_of=hubert_kmeans.seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )
        self.print(
            f'training on {feature_extraction_num_steps * feature_extraction_batch_size} out of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=feature_extraction_batch_size, shuffle=True)

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

    def print(self, msg):
        print(msg)

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_hubert_features(self):

        raw_wave = next(self.dl_iter)[0]

        embed = self.hubert_kmeans.forward(wav_input=raw_wave.to(self.device), return_embed=True)

        # get features
        embed = rearrange(embed, 'b t f -> (b t) f')
        embed = embed.detach().cpu().numpy()

        return embed

    def train(self, log_fn=noop, seed=0):

        self.print('step 1: extracting features. must wait for this to complete before training kmeans.')
        features = []
        while self.steps < self.feature_extraction_num_steps:
            self.print(f'{int(self.steps.item())} / {self.feature_extraction_num_steps} steps')
            features.append(self.extract_hubert_features())
            self.steps += 1

        features = np.concatenate(features, axis=0)

        features = features[~np.any(np.isnan(features), axis=-1)]

        self.print('step 2: training kmeans')
        learn_kmeans(features, seed, str(self.results_folder / 'kmeans.joblib'), n_clusters=self.hubert_kmeans.codebook_size)

        self.print('training complete')