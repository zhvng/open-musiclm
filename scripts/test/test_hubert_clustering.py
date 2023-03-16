import os
import sys

import torch
import torchaudio
from torchaudio.functional import resample
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from open_musiclm.config import load_model_config, load_training_config, create_hubert_kmeans_from_config, create_hubert_kmeans_trainer_from_config
from open_musiclm.utils import zero_mean_unit_var_norm, int16_to_float32, float32_to_int16, exists
from open_musiclm.open_musiclm import get_or_compute_semantic_token_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test hubert kmeans to see the difference in sequences')
    parser.add_argument('--model_config', default='./configs/model/musiclm_small.json')
    parser.add_argument('--kmeans_path', default='./results/hubert_kmeans/kmeans.joblib')
    parser.add_argument('--folder', default='./data/fma_large')

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading hubert...')
    wav2vec = create_hubert_kmeans_from_config(model_config, args.kmeans_path, device)
    
    path = Path(args.folder)
    assert path.exists(), 'folder does not exist'

    files = []
    for ext in ['mp3', 'wav', 'flac']:
        for file in path.glob(f'**/*.{ext}'):
            files.append(file)
    assert len(files) > 0, 'no sound files found'

    start_audio = 20000
    audio_lengths = [4, 10, 15, 25]
    batch_size = 16
    shortest_length = None
    cropped_semantic_tokens_for_each_length = []
    for audio_seconds in audio_lengths:
        audios_for_wav2vec = []
        for audio_path in files[start_audio: start_audio + 16]:
            data, sample_hz = torchaudio.load(audio_path)

            if data.shape[0] > 1:
                data = torch.mean(data, dim=0).unsqueeze(0)

            target_length = int(audio_seconds * sample_hz)
            normalized_data = zero_mean_unit_var_norm(data)

            normalized_data = normalized_data[: , :target_length]

            audio_for_wav2vec = resample(normalized_data, sample_hz, wav2vec.target_sample_hz)

            audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))

            audios_for_wav2vec.append(audio_for_wav2vec)
        
        audios_for_wav2vec = torch.cat(audios_for_wav2vec, dim=0).to(device)
        semantic_token_ids = get_or_compute_semantic_token_ids(None, audios_for_wav2vec, wav2vec)
        print(semantic_token_ids.shape)

        if not exists(shortest_length):
            shortest_length = semantic_token_ids.shape[1]
        else:
            l = semantic_token_ids.shape[1]
            if l < shortest_length:
                shortest_length = l

        cropped_semantic_tokens_for_each_length.append(semantic_token_ids[:, :shortest_length])

    print(cropped_semantic_tokens_for_each_length[0][0])
    print(cropped_semantic_tokens_for_each_length[1][0])
    # get accuracy compared to last elem in cropped_semantic_tokens_for_each_length

    side_length = len(cropped_semantic_tokens_for_each_length)
    accuracy_matrix = np.zeros((side_length, side_length))

    for i in range(side_length):
        for j in range(side_length):
            accuracy = torch.mean((cropped_semantic_tokens_for_each_length[i] == cropped_semantic_tokens_for_each_length[j]).float())
            print(f'% similar between {audio_lengths[i]} and {audio_lengths[j]} second audio: {accuracy}')
            accuracy_matrix[i][j] = accuracy

    # plot the accuracy matrix in a grid with a title and axis labels

    # create a heatmap with darker colors representing higher accuracy
    fig, ax = plt.subplots()
    im = ax.imshow(accuracy_matrix, cmap='Blues', vmin=0, vmax=1)

    # remove ticks from the plot
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # move the x-axis ticks to the top of the grid
    ax.xaxis.tick_top()

    # add a colorbar legend
    cbar = ax.figure.colorbar(im, ax=ax)

    # set axis labels
    ax.set_xticks(np.arange(len(audio_lengths)))
    ax.set_yticks(np.arange(len(audio_lengths)))
    ax.set_xticklabels(audio_lengths)
    ax.set_yticklabels(audio_lengths)

    # add text annotations for each cell
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, round(accuracy_matrix[i, j], 2),
                        ha="center", va="center", color="w")

    # set plot title
    ax.set_title("Semantic Token Similarity Between Various Total Audio Lengths")
    ax.set_xlabel("total audio length (seconds)")
    ax.set_ylabel("total audio length (seconds)")

    # show the plot
    plt.savefig('./results/accuracy_matrix.png')
 

