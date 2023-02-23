import os
import sys

import torch


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import HubertModel, HubertPreTrainedModel
from open_musiclm.hf_hubert_kmeans import HfHubertWithKmeans, get_hubert_kmeans
from open_musiclm.trainer import HfHubertKmeansTrainer
from scripts.train_utils import disable_print

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_folder = './data/fma_large'

print('loading hubert...')
hubert_kmeans = get_hubert_kmeans(model_name='m-a-p/MERT-v0', kmeans_path=None)
trainer = HfHubertKmeansTrainer(
    feature_extraction_num_steps=100,
    feature_extraction_batch_size=64,
    data_max_length_seconds=1,
    hubert_kmeans=hubert_kmeans,
    folder=audio_folder,
    results_folder='./results/hubert_kmeans',
).to(device)

trainer.train()