import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_encodec_from_config,
                                 create_hubert_kmeans_from_config,
                                 load_model_config, load_training_config,
                                 create_data_preprocessor_from_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train coarse stage')
    parser.add_argument('--stage', default='all', help='Name of the stage we want to process data for (semantic, coarse, fine, all). Provide "all" to process all stages in succession.')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_fma_preprocess.json')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')

    args = parser.parse_args()

    print(f'using model config {args.model_config}, training config {args.training_config}, rvq checkpoint {args.rvq_path}, kmeans checkpoint {args.kmeans_path}')
    print(f'processing stage(s): {args.stage}')

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_samples = training_config.data_preprocessor_cfg.batch_size * training_config.data_preprocessor_cfg.batches_per_shard * training_config.data_preprocessor_cfg.num_shards
    print(f'you will be preprocessing enough data for {total_samples} unique training samples. Make sure this is enough. This may take a while...')

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

    print('loading wav2vec...')
    wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)

    processor = create_data_preprocessor_from_config(model_config, training_config, clap, wav2vec, encodec_wrapper, args.stage, device)

    processor.process()

