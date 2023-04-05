import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from beartype import beartype
from beartype.typing import Literal, Optional, List

from .clap_quantized import ClapQuantized, create_clap_quantized
from .encodec_wrapper import EncodecWrapper, create_encodec_24khz
from .hf_hubert_kmeans import HfHubertWithKmeans, get_hubert_kmeans
from .open_musiclm import (MusicLM, TokenConditionedTransformer,
                           create_coarse_transformer, create_fine_transformer,
                           create_semantic_transformer)
from .trainer import ClapRVQTrainer, HfHubertKmeansTrainer, SingleStageTrainer
from .preprocess import DataPreprocessor
from .utils import exists


@dataclass
class ClapRVQConfig:
    rq_num_quantizers: int
    codebook_size: int
    enable_fusion: bool = False
    rq_ema_decay: float = 0.95
    threshold_ema_dead_code: float = 0.0

@dataclass
class HubertKmeansConfig:
    model_name: str
    normalize_embeds: bool
    embed_layer: int = 7
    target_sample_hz: int = 16000
    seq_len_multiple_of: int = 320
    codebook_size: int = 1024
    output_hz: int = 50

@dataclass
class EncodecConfig:
    bandwidth: float
    codebook_size: int
    output_hz: int = 75

RelativePositionBiasType = Literal['continuous', 't5']

@dataclass
class SemanticConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = True

@dataclass
class CoarseConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = True

@dataclass
class FineConfig:
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    grad_shrink_alpha: float = 0.1
    non_causal_prefix_size: int = 0
    relative_position_bias_type: RelativePositionBiasType = 'continuous'
    use_memory_efficient_attention: bool = True

@dataclass
class GlobalConfig:
    semantic_audio_length_seconds: float = 10.0
    coarse_audio_length_seconds: float = 4.0
    fine_audio_length_seconds: float = 2.0
    clap_audio_length_seconds: float = 10.0
    num_coarse_quantizers: int = 3
    num_fine_quantizers: int = 5

@beartype
@dataclass
class MusicLMModelConfig:
    clap_rvq_cfg: ClapRVQConfig
    hubert_kmeans_cfg: HubertKmeansConfig
    encodec_cfg: EncodecConfig
    semantic_cfg: SemanticConfig
    coarse_cfg: CoarseConfig
    fine_cfg: FineConfig
    global_cfg: GlobalConfig


@dataclass
class ClapRVQTrainerConfig:
    folder: str
    num_train_steps: int
    batch_size: int
    accumulate_batches: int
    save_model_every: int
    save_results_every: int

@dataclass
class HubertKmeansTrainerConfig:
    folder: str
    feature_extraction_num_steps: int
    feature_extraction_batch_size: int

@dataclass
class SingleStageTrainerConfig:
    stage: str
    folder: str
    valid_frac: float
    lr: float
    lr_warmup: int
    batch_size: int
    grad_accum_every: int
    wd: float
    max_grad_norm: float
    cross_entropy_loss_weights: list[float]
    num_train_steps: int
    save_results_every: int
    save_model_every: int
    save_predicted_tokens: bool
    save_reconstructed_wave: bool
    use_preprocessed_data: bool

@dataclass
class DataPreprocessorConfig:
    folder: str = './data/fma_large'
    results_folder: str = './fma_preprocessed'
    max_audio_length_seconds: int = 30
    random_crop: bool = True
    clap_batch_size: int = 32

@beartype
@dataclass
class MusicLMTrainingConfig:
    clap_rvq_trainer_cfg: ClapRVQTrainerConfig
    hubert_kmeans_trainer_cfg: HubertKmeansTrainerConfig
    semantic_trainer_cfg: SingleStageTrainerConfig
    coarse_trainer_cfg: SingleStageTrainerConfig
    fine_trainer_cfg: SingleStageTrainerConfig
    data_preprocessor_cfg: DataPreprocessorConfig
    

@beartype
def load_model_config(config_path: str) -> MusicLMModelConfig:
    with open(config_path, 'r') as f:
        config = json.load(f)

    return MusicLMModelConfig(
        clap_rvq_cfg=ClapRVQConfig(**config['clap_rvq_cfg']),
        hubert_kmeans_cfg=HubertKmeansConfig(**config['hubert_kmeans_cfg']),
        encodec_cfg=EncodecConfig(**config['encodec_cfg']),
        semantic_cfg=SemanticConfig(**config['semantic_cfg']),
        coarse_cfg=CoarseConfig(**config['coarse_cfg']),
        fine_cfg=FineConfig(**config['fine_cfg']),
        global_cfg=GlobalConfig(**config['global_cfg']),
    )

@beartype
def load_training_config(config_path: str) -> MusicLMTrainingConfig:
    with open(config_path, 'r') as f:
        config = json.load(f)

    return MusicLMTrainingConfig(
        clap_rvq_trainer_cfg=ClapRVQTrainerConfig(**config['clap_rvq_trainer_cfg']),
        hubert_kmeans_trainer_cfg=HubertKmeansTrainerConfig(**config['hubert_kmeans_trainer_cfg']),
        semantic_trainer_cfg=SingleStageTrainerConfig(**config['semantic_trainer_cfg']),
        coarse_trainer_cfg=SingleStageTrainerConfig(**config['coarse_trainer_cfg']),
        fine_trainer_cfg=SingleStageTrainerConfig(**config['fine_trainer_cfg']),
        data_preprocessor_cfg=DataPreprocessorConfig(**config['data_preprocessor_cfg']),
    )

# helper functions

def load_model(model, path):
    """helper class to load a model checkpoint"""
    path = Path(path)
    assert path.exists(), f'checkpoint does not exist at {str(path)}'
    pkg = torch.load(str(path))
    model.load_state_dict(pkg)

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# model stages

@beartype
def create_clap_quantized_from_config(model_config: MusicLMModelConfig, rvq_path: Optional[str], device) -> ClapQuantized:
    with disable_print():
        return create_clap_quantized(
            **asdict(model_config.clap_rvq_cfg),
            device=device,
            learn_rvq=False,
            rvq_checkpoint_path=rvq_path,
        ).to(device)

@beartype
def create_hubert_kmeans_from_config(model_config: MusicLMModelConfig, kmeans_path: Optional[str], device) -> HfHubertWithKmeans:
    return get_hubert_kmeans(
        **asdict(model_config.hubert_kmeans_cfg),
        kmeans_path=kmeans_path,
    ).to(device)

@beartype
def create_encodec_from_config(model_config: MusicLMModelConfig, device) -> EncodecWrapper:
    return create_encodec_24khz(**asdict(model_config.encodec_cfg)).to(device)

@beartype
def create_semantic_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device
) -> TokenConditionedTransformer:
    transformer = create_semantic_transformer(
        **asdict(model_config.semantic_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

@beartype
def create_coarse_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device
) -> TokenConditionedTransformer:
    transformer = create_coarse_transformer(
        **asdict(model_config.coarse_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        semantic_codebook_size=model_config.hubert_kmeans_cfg.codebook_size,
        acoustic_codebook_size=model_config.encodec_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
        num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

@beartype
def create_fine_transformer_from_config(
    model_config: MusicLMModelConfig,
    checkpoint_path: Optional[str],
    device
) -> TokenConditionedTransformer:
    transformer = create_fine_transformer(
        **asdict(model_config.fine_cfg),
        clap_codebook_size=model_config.clap_rvq_cfg.codebook_size,
        acoustic_codebook_size=model_config.encodec_cfg.codebook_size,
        num_clap_quantizers=model_config.clap_rvq_cfg.rq_num_quantizers,
        num_coarse_quantizers=model_config.global_cfg.num_coarse_quantizers,
        num_fine_quantizers=model_config.global_cfg.num_fine_quantizers,
    ).to(device)

    if exists(checkpoint_path):
        load_model(transformer, checkpoint_path)

    return transformer

# trainers

@beartype
def create_clap_rvq_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    clap: ClapQuantized,
    results_folder: str,
    device,
    accelerate_kwargs: dict = {},
    config_paths: Optional[List[str]] = None
):
    trainer = ClapRVQTrainer(
        audio_conditioner=clap,
        results_folder=results_folder,
        data_max_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        accelerate_kwargs=accelerate_kwargs,
        config_paths=config_paths,
        **asdict(training_config.clap_rvq_trainer_cfg)
    ).to(device)

    return trainer

@beartype
def create_hubert_kmeans_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    hubert_kmeans: HfHubertWithKmeans,
    results_folder: str,
    device,
    config_paths: Optional[List[str]] = None
):
    trainer = HfHubertKmeansTrainer(
        hubert_kmeans=hubert_kmeans,
        results_folder=results_folder,
        data_max_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        config_paths=config_paths,
        **asdict(training_config.hubert_kmeans_trainer_cfg),
    ).to(device)

    return trainer

@beartype
def create_single_stage_trainer_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    stage: Literal['semantic', 'coarse', 'fine'],
    results_folder: str,
    transformer: TokenConditionedTransformer,
    clap: Optional[ClapQuantized]=None,
    wav2vec: Optional[HfHubertWithKmeans]=None,
    encodec_wrapper: Optional[EncodecWrapper]=None,
    device='cpu',
    accelerate_kwargs: dict = {},
    config_paths: Optional[List[str]] = None
) -> SingleStageTrainer:
    
    semantic_audio_length_seconds = model_config.global_cfg.semantic_audio_length_seconds
    coarse_audio_length_seconds = model_config.global_cfg.coarse_audio_length_seconds
    fine_audio_length_seconds = model_config.global_cfg.fine_audio_length_seconds

    if stage == 'semantic':
        trainer_cfg = training_config.semantic_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, semantic_audio_length_seconds)
    elif stage == 'coarse':
        trainer_cfg = training_config.coarse_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, coarse_audio_length_seconds, coarse_audio_length_seconds)
    elif stage == 'fine':
        trainer_cfg = training_config.fine_trainer_cfg
        data_max_length_seconds = (semantic_audio_length_seconds, fine_audio_length_seconds)
    
    trainer = SingleStageTrainer(
        model_config=model_config,
        training_config=training_config,
        transformer=transformer,
        audio_conditioner=clap,
        wav2vec=wav2vec,
        neural_codec=encodec_wrapper,
        results_folder=results_folder,
        data_max_length_seconds=data_max_length_seconds,
        accelerate_kwargs=accelerate_kwargs,
        config_paths=config_paths,
        **asdict(trainer_cfg)
    ).to(device)

    return trainer

@beartype
def create_data_preprocessor_from_config(
    model_config: MusicLMModelConfig,
    training_config: MusicLMTrainingConfig,
    clap: ClapQuantized,
    wav2vec: HfHubertWithKmeans,
    encodec_wrapper: EncodecWrapper,
    device='cpu',
    config_paths: Optional[List[str]] = None
):
    data_preprocessor = DataPreprocessor(
        audio_conditioner=clap,
        wav2vec=wav2vec,
        neural_codec=encodec_wrapper,
        semantic_audio_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        coarse_audio_length_seconds=model_config.global_cfg.coarse_audio_length_seconds,
        fine_audio_length_seconds=model_config.global_cfg.fine_audio_length_seconds,
        clap_audio_length_seconds=model_config.global_cfg.clap_audio_length_seconds,
        config_paths=config_paths,
        **asdict(training_config.data_preprocessor_cfg)
    ).to(device)

    return data_preprocessor

# entire model

@beartype
def create_musiclm_from_config(
    model_config: MusicLMModelConfig,
    semantic_path: str,
    coarse_path: str,
    fine_path: str,
    rvq_path: str,
    kmeans_path: str,
    device
):
    clap = create_clap_quantized_from_config(model_config, rvq_path, device)
    wav2vec = create_hubert_kmeans_from_config(model_config, kmeans_path, device)
    encodec_wrapper = create_encodec_from_config(model_config, device)
    semantic_transformer = create_semantic_transformer_from_config(model_config, semantic_path, device)
    coarse_transformer = create_coarse_transformer_from_config(model_config, coarse_path, device)
    fine_transformer = create_fine_transformer_from_config(model_config, fine_path, device)

    musiclm = MusicLM(
        wav2vec=wav2vec,
        clap=clap,
        neural_codec=encodec_wrapper,
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        fine_transformer=fine_transformer,
    ).to(device)

    return musiclm
