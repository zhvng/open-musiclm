import os
import sys

import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import load_model_config, load_training_config, create_hubert_kmeans_from_config, create_hubert_kmeans_trainer_from_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train kmeans to quantize hubert embeddings')
    parser.add_argument('--results_folder', default='./results/hubert_kmeans')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_musiclm_fma.json')

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading hubert...')
    hubert_kmeans = create_hubert_kmeans_from_config(model_config, None, device)

    trainer = create_hubert_kmeans_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        hubert_kmeans=hubert_kmeans,
        results_folder=args.results_folder,
        device=device,
        config_paths=[args.model_config, args.training_config]
    )

    trainer.train()