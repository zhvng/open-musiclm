import json
import os
from dataclasses import dataclass
from beartype import beartype

@dataclass
class ClapRVQConfig:
    checkpoint_path: str
    rq_num_quantizers: int

@dataclass
class HubertKmeansConfig:
    model_name: str
    normalize_input: bool
    normalize_embeds: bool

@dataclass
class EncodecConfig:
    bandwidth: float

@dataclass
class SemanticConfig:
    dim: int
    depth: int
    num_clap_quantizers: int

@dataclass
class CoarseConfig:
    dim: int
    depth: int
    num_clap_quantizers: int
    num_coarse_quantizers: int

@dataclass
class FineConfig:
    dim: int
    depth: int
    num_clap_quantizers: int
    num_coarse_quantizers: int
    num_fine_quantizers: int

@dataclass
class GlobalConfig:
    semantic_audio_length_seconds: float
    coarse_audio_length_seconds: float
    fine_audio_length_seconds: float

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
    data_max_length_seconds: float

@dataclass
class SingleStageTrainerConfig:
    stage: str
    folder: str
    valid_frac: float
    lr: float
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

@beartype
@dataclass
class MusicLMTrainingConfig:
    clap_rvq_trainer_cfg: ClapRVQTrainerConfig
    hubert_kmeans_trainer_cfg: HubertKmeansTrainerConfig
    semantic_trainer_cfg: SingleStageTrainerConfig
    coarse_trainer_cfg: SingleStageTrainerConfig
    fine_trainer_cfg: SingleStageTrainerConfig
    

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
    )
