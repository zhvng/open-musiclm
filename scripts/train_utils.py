from functools import wraps
import logging
import sys
import os
from pathlib import Path

def exists(val):
    return val is not None

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_latest_checkpoints(results_folder, max_step=None):
    highest_transformer_step = -1
    highest_optimizer_step = -1
    highest_scheduler_step = -1
    transformer_path = None
    optimizer_path = None
    scheduler_path = None
    max_step = float('inf') if max_step is None else max_step
    for file in os.listdir(results_folder):
        if file.endswith('.pt'):
            if 'transformer' in file:
                step = int(file.split('.')[2])
                if step > highest_transformer_step and step <= max_step:
                    highest_transformer_step = step
                    transformer_path = os.path.join(results_folder, file)
            elif 'optimizer' in file:
                step = int(file.split('.')[2])
                if step > highest_optimizer_step and step <= max_step:
                    highest_optimizer_step = step
                    optimizer_path = os.path.join(results_folder, file)
            elif 'scheduler' in file:
                step = int(file.split('.')[2])
                if step > highest_scheduler_step and step <= max_step:
                    highest_scheduler_step = step
                    scheduler_path = os.path.join(results_folder, file)

    assert highest_transformer_step == highest_optimizer_step, 'transformer and optimizer checkpoints are not aligned'
    if scheduler_path is not None:
        assert highest_transformer_step == highest_scheduler_step, 'transformer and scheduler checkpoints are not aligned'

    return (transformer_path, optimizer_path, scheduler_path), highest_transformer_step

def validate_train_args(args):
    assert not(exists(args.fine_tune_from) and exists(args.continue_from_dir)), 'choose one: fine tune from a checkpoint or continue from a directory'

    print(f'saving results to {args.results_folder}, using model config {args.model_config} and training config {args.training_config}, using rvq checkpoint {args.rvq_path} and kmeans checkpoint {args.kmeans_path}')
    if exists(args.continue_from_dir):
        print(f'continuing from latest checkpoint in {args.continue_from_dir}')
        assert not Path(args.continue_from_dir) == Path(args.results_folder), 'continue_from_dir must be different from results_folder'
    elif exists(args.fine_tune_from):
        print(f'fine tuning from checkpoint {args.fine_tune_from}. Make sure to use the same model config as the base model.')

def load_checkpoint_from_args(trainer, args):
    if exists(args.continue_from_dir):
        checkpoints, steps = get_latest_checkpoints(args.continue_from_dir, args.continue_from_step)
        print(f'loading checkpoints: {checkpoints}')
        trainer.load(*checkpoints, steps=steps+1)
