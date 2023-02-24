import os
import sys

import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.clap_quantized import create_clap_quantized
from open_musiclm.open_musiclm import create_fine_transformer 
from open_musiclm.encodec_wrapper import create_encodec_24khz
from open_musiclm.trainer import SingleStageTrainer
from scripts.train_utils import disable_print, get_latest_checkpoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train fine stage')
    parser.add_argument('--results_folder', default='./results/fine')
    parser.add_argument('--audio_folder', default='./data/fma_large')
    parser.add_argument('--continue_from_dir', default=None, type=str)
    args = parser.parse_args()

    audio_folder = args.audio_folder
    results_folder = args.results_folder

    print(f'training on {audio_folder} and saving results to {results_folder}')
    if args.continue_from_dir is not None:
        print(f'continuing from latest checkpoint in {args.continue_from_dir}')
        assert not os.path.samefile(args.continue_from_dir, results_folder), 'continue_from_dir must be different from results_folder'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading clap...')
    clap_checkpoint = "./checkpoints/clap-laion-audioset-fusion.pt"
    rvq_checkpoint = './checkpoints/clap.rvq.350.pt'
    with disable_print():
        clap = create_clap_quantized(device=device, learn_rvq=False, checkpoint_path=clap_checkpoint, rvq_checkpoint_path=rvq_checkpoint).to(device)

    print('loading encodec')
    encodec_wrapper = create_encodec_24khz(bandwidth=6.).to(device)

    # 8 tokens per timestep @ 75 Hz
    # lets do 3 coarse 5 fine

    print('loading fine stage...')
    fine_transformer = create_fine_transformer(
        dim=1024,
        depth=6,
        clap_codebook_size=clap.codebook_size,
        acoustic_codebook_size=encodec_wrapper.codebook_size,
        num_coarse_quantizers=3,
        num_fine_quantizers=5,
    ).to(device)

    trainer = SingleStageTrainer(
        transformer=fine_transformer,
        stage='fine',
        audio_conditioner=clap,
        neural_codec=encodec_wrapper,
        folder=audio_folder,
        lr=3e-4,
        batch_size=2,
        grad_accum_every=8,
        data_max_length_seconds=2,
        num_train_steps=15001,
        save_results_every=500,
        results_folder=results_folder,
        accelerate_kwargs={
            'log_with': "tensorboard",
            'logging_dir': './logs/fine'
        }
    ).to(device)

    if args.continue_from_dir is not None:
        transformer_checkpoint, optimizer_checkpoint = get_latest_checkpoints(args.continue_from_dir)
        print(f'loading checkpoint {transformer_checkpoint} and {optimizer_checkpoint}')
        trainer.load(transformer_checkpoint, optimizer_checkpoint)

    print('training!')
    trainer.train()
