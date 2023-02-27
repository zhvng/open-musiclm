import os
import sys

import torch
from dataclasses import asdict
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import HubertModel, HubertPreTrainedModel
from open_musiclm.hf_hubert_kmeans import HfHubertWithKmeans, get_hubert_kmeans
from open_musiclm.trainer import HfHubertKmeansTrainer
from open_musiclm.config import load_model_config, load_training_config
from scripts.train_utils import disable_print

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
    hubert_kmeans = get_hubert_kmeans(
        kmeans_path=None,
        **asdict(model_config.hubert_kmeans_cfg),
    ).to(device)

    trainer = HfHubertKmeansTrainer(
        hubert_kmeans=hubert_kmeans,
        results_folder='./results/hubert_kmeans_normalize_inputs_no_normalize_embeds_8',
        **asdict(training_config.hubert_kmeans_trainer_cfg),
    ).to(device)

    trainer.train()