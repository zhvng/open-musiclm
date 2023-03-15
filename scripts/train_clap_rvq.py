import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_clap_rvq_trainer_from_config,
                                 load_model_config, load_training_config)
from open_musiclm.trainer import ClapRVQTrainer
from scripts.train_utils import disable_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train rvq to quantize clap embeddings')
    parser.add_argument('--results_folder', default='./results/clap_rvq')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')
    parser.add_argument('--continue_from', default=None, type=str)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, args.continue_from, device)

    trainer = create_clap_rvq_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        clap=clap,
        results_folder=args.results_folder,
        device=device,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/clap_rvq'
        },
        config_paths=[args.model_config, args.training_config])

    trainer.train()