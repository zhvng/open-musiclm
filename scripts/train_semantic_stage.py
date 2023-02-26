import os
import sys

import torch
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_semantic_transformer
from open_musiclm.trainer import SingleStageTrainer
from open_musiclm.hf_hubert_kmeans import get_hubert_kmeans
from scripts.train_utils import disable_print, get_latest_checkpoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train semantic stage')
    parser.add_argument('--results_folder', default='./results/semantic')
    parser.add_argument('--audio_folder', default='./data/fma_large')
    parser.add_argument('--continue_from_dir', default=None, type=str)
    args = parser.parse_args()

    audio_folder = args.audio_folder
    results_folder = args.results_folder

    print(f'training on {audio_folder} and saving results to {results_folder}')
    if args.continue_from_dir is not None:
        print(f'continuing from latest checkpoint in {args.continue_from_dir}')
        assert not Path(args.continue_from_dir) == Path(results_folder), 'continue_from_dir must be different from results_folder'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
    rvq_checkpoint = './checkpoints/clap.rvq.350.pt'
    with disable_print():
        clap = create_clap_quantized(device=device, learn_rvq=False, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=rvq_checkpoint).to(device)

    print('loading wav2vec...')
    wav2vec = get_hubert_kmeans(
        model_name='m-a-p/MERT-v0', 
        kmeans_path='./results/hubert_kmeans/kmeans.joblib',
        normalize_input=True,
        normalize_embeds=True,
    ).to(device)

    print('loading semantic stage...')
    semantic_transformer = create_semantic_transformer(
        dim=1024,
        depth=6,
        clap_codebook_size=clap.codebook_size,
        semantic_codebook_size=wav2vec.codebook_size,
    ).to(device)

    trainer = SingleStageTrainer(
        transformer=semantic_transformer,
        stage='semantic',
        audio_conditioner=clap,
        wav2vec=wav2vec,
        folder=audio_folder,
        batch_size=4,
        grad_accum_every=8,
        data_max_length_seconds=8,
        num_train_steps=15001,
        results_folder=results_folder,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/semantic'
        }
    ).to(device)

    if args.continue_from_dir is not None:
        transformer_checkpoint, optimizer_checkpoint = get_latest_checkpoints(args.continue_from_dir)
        print(f'loading checkpoint {transformer_checkpoint} and {optimizer_checkpoint}')
        trainer.load(transformer_checkpoint, optimizer_checkpoint)

    print('training!')
    trainer.train()
