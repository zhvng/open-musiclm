import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from open_musiclm.config import load_model_config, load_training_config


model_config = load_model_config('./configs/model/musiclm_small.json')
print(model_config)

training_config = load_training_config('./configs/training/train_musiclm_fma.json')
print(training_config)

print('\nok!')