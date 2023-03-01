from functools import wraps
import logging
import sys
import os
from audiolm_pytorch import HubertWithKmeans

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_latest_checkpoints(results_folder):
    highest_transformer_step = -1
    highest_optimizer_step = -1
    highest_scheduler_step = -1
    transformer_path = None
    optimizer_path = None
    scheduler_path = None
    for file in os.listdir(results_folder):
        if file.endswith('.pt'):
            if 'transformer' in file:
                step = int(file.split('.')[2])
                if step > highest_transformer_step:
                    highest_transformer_step = step
                    transformer_path = os.path.join(results_folder, file)
            elif 'optimizer' in file:
                step = int(file.split('.')[2])
                if step > highest_optimizer_step:
                    highest_optimizer_step = step
                    optimizer_path = os.path.join(results_folder, file)
            elif 'scheduler' in file:
                step = int(file.split('.')[2])
                if step > highest_scheduler_step:
                    highest_scheduler_step = step
                    scheduler_path = os.path.join(results_folder, file)

    assert highest_transformer_step == highest_optimizer_step, 'transformer and optimizer checkpoints are not aligned'
    if scheduler_path is not None:
        assert highest_transformer_step == highest_scheduler_step, 'transformer and scheduler checkpoints are not aligned'

    return transformer_path, optimizer_path, scheduler_path
