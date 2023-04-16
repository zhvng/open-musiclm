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
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--training_config', default='./configs/training/train_fma_preprocess.json')
    parser.add_argument('--rvq_path', default='./checkpoints/clap.rvq.350.pt')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--filter_fma', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    print(f'using model config {args.model_config}, training config {args.training_config}, rvq checkpoint {args.rvq_path}, kmeans checkpoint {args.kmeans_path}')

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap = create_clap_quantized_from_config(model_config, args.rvq_path, device)

    print('loading wav2vec...')
    wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)

    print('loading encodec...')
    encodec_wrapper = create_encodec_from_config(model_config, device)


    # get rid of some experimental tracks, see notebooks/analyze_fma.ipynb
    if args.filter_fma:
        try:
            import pandas as pd
            import ast
        except ImportError:
            pd = None

        assert pd is not None, 'pandas not found, please install pandas to filter fma'

        metadata_folder = training_config.data_preprocessor_cfg.metadata_folder

        tracks = pd.read_csv(os.path.join(metadata_folder, 'tracks.csv'), index_col=0, header=[0, 1])
        experimental_genre = 38
        experimental_tracks = tracks.loc[tracks['track', 'genres_all'].apply(lambda x: experimental_genre in ast.literal_eval(x))]
        ignore_files = list(experimental_tracks.loc[(experimental_tracks['track', 'listens'] <= 1000) | (experimental_tracks['track', 'favorites'] <= 5)].index)
        ignore_files = [f'{i:06d}.mp3' for i in ignore_files]
    else:
        ignore_files = None

    processor = create_data_preprocessor_from_config(
        model_config,
        training_config,
        clap,
        wav2vec,
        encodec_wrapper,
        device,
        config_paths=[args.model_config, args.training_config],
        ignore_files=ignore_files)

    processor.process()

