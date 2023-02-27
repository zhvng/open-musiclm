import os
import sys

import torch
import argparse
from pathlib import Path
from dataclasses import asdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_fine_transformer 
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.trainer import SingleStageTrainer
from open_musiclm.config import load_model_config, load_training_config
from scripts.train_utils import disable_print, get_latest_checkpoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train fine stage')
    parser.add_argument('--results_folder', default='./results/fine')
    parser.add_argument('--continue_from_dir', default=None, type=str)
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')

    args = parser.parse_args()

    print(f'saving results to {args.results_folder}, using model config {args.model_config} and training config {args.training_config}, using rvq checkpoint {args.rvq_path} and kmeans checkpoint {args.kmeans_path}')
    if args.continue_from_dir is not None:
        print(f'continuing from latest checkpoint in {args.continue_from_dir}')
        assert not Path(args.continue_from_dir) == Path(args.results_folder), 'continue_from_dir must be different from results_folder'

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    with disable_print():
        clap = create_clap_quantized(
            device=device, 
            learn_rvq=False, 
            rvq_checkpoint_path=args.rvq_path,
            **asdict(model_config.clap_rvq_cfg),
        ).to(device)
        
    print('loading encodec')
    encodec_wrapper = create_encodec_24khz(**asdict(model_config.encodec_cfg)).to(device)

    # 8 tokens per timestep @ 75 Hz
    # lets do 3 coarse 5 fine

    print('loading fine stage...')
    fine_transformer = create_fine_transformer(
        clap_codebook_size=clap.codebook_size,
        acoustic_codebook_size=encodec_wrapper.codebook_size,
        **asdict(model_config.fine_cfg)
    ).to(device)

    trainer = SingleStageTrainer(
        transformer=fine_transformer,
        audio_conditioner=clap,
        neural_codec=encodec_wrapper,
        data_max_length_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        results_folder=args.results_folder,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/fine'
        },
        **asdict(training_config.fine_trainer_cfg)
    ).to(device)

    if args.continue_from_dir is not None:
        transformer_checkpoint, optimizer_checkpoint = get_latest_checkpoints(args.continue_from_dir)
        print(f'loading checkpoint {transformer_checkpoint} and {optimizer_checkpoint}')
        trainer.load(transformer_checkpoint, optimizer_checkpoint)

    print('training!')
    trainer.train()
