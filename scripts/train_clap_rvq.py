import os
import sys

import torch
from dataclasses import asdict
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.trainer import ClapRVQTrainer
from open_musiclm.config import load_model_config, load_training_config
from scripts.train_utils import disable_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train rvq to quantize clap embeddings')
    parser.add_argument('--results_folder', default='./results/clap_rvq')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with disable_print():
        clap = create_clap_quantized(
            device=device, 
            learn_rvq=True,
            rvq_checkpoint_path=None,
            **asdict(model_config.clap_rvq_cfg),
        ).to(device)

    trainer = ClapRVQTrainer(
        audio_conditioner=clap,
        results_folder=args.results_folder,
        **asdict(training_config.clap_rvq_trainer_cfg),
    ).to(device)

    trainer.train()