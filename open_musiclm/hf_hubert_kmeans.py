from pathlib import Path

import torch
from torch import nn
import numpy as np
from einops import reduce, pack, unpack
from beartype.typing import Optional
import math

from torchaudio.functional import resample
from .utils import exists, curtail_to_multiple, zero_mean_unit_var_norm
from transformers import HubertModel
from sklearn.cluster import MiniBatchKMeans

import joblib
import logging
logging.root.setLevel(logging.ERROR)


class HfHubertWithKmeans(nn.Module):
    """
    Hugging Face HubertModel + a k-means layer on top. Pretrained checkpoint for music: https://huggingface.co/m-a-p/MERT-v0
    Note: MERT-v0 outputs features at 50Hz while Wav2Vec-BERT (used in the paper) outputs at 25 Hz.
          We can reduce the number of semantic tokens by averaging adjacent features.
    """

    def __init__(
        self,
        *,
        hubert: HubertModel,
        kmeans: Optional[MiniBatchKMeans] = None,
        embed_layer: int=7,
        target_sample_hz=16000,
        seq_len_multiple_of=int(16000 / 50),
        normalize_input=True,
        normalize_embeds=True,
        codebook_size: int=1024,
        output_hz: int=50,
        context_window_seconds: Optional[float]=None,
        bin_size: int=1
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.codebook_size = kmeans.n_clusters if exists(kmeans) else None
        self.context_window_seconds = context_window_seconds
        self.bin_size = bin_size
        self.output_hz = output_hz

        self.codebook_size = codebook_size
        if exists(kmeans):
            assert self.codebook_size == kmeans.n_clusters, "codebook_size must match kmeans.n_clusters"

        self.normalize_input = normalize_input
        self.normalize_embeds = normalize_embeds

        self.embed_layer = embed_layer

        self.hubert = hubert
        self.kmeans = kmeans

    @torch.no_grad()
    def forward(
        self,
        wav_input: torch.Tensor,
        flatten=True,
        return_embed=False,
        input_sample_hz=None
    ):
        assert return_embed or exists(self.kmeans), "kmeans model must be provided if return_embed==False"

        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.context_window_seconds):
            target_length = int(self.context_window_seconds * self.target_sample_hz)
            wav_input = list(wav_input.split(target_length, dim=-1))
            wav_input[-1] = torch.nn.functional.pad(wav_input[-1], (0, target_length - wav_input[-1].shape[-1]))
            wav_input, packed_wav_input_shape = pack(wav_input, '* d')
        else:
            packed_wav_input_shape = None

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        if self.normalize_input:
            wav_input = zero_mean_unit_var_norm(wav_input)

        hubert_args = {
            'input_values': wav_input,
            'attention_mask': torch.ones_like(wav_input, device=device), # TODO: handle padding
        }

        outputs = self.hubert(**hubert_args, output_hidden_states = True)
        embed = outputs.hidden_states[self.embed_layer]

        # pad and reduce with bin size
        audio_length_seconds = wav_input.shape[-1] / self.target_sample_hz
        pad_to = int(self.output_hz * self.bin_size * audio_length_seconds)
        if embed.shape[1] < pad_to:
            # repeat last few frames
            embed = torch.cat([embed, embed[:, -1:, :].repeat(1, pad_to - embed.shape[1], 1)], dim=1)
        if self.bin_size > 1:
            embed = reduce(embed, '... (n n1) f -> ... n f', reduction='mean', n1=self.bin_size)

        if exists(packed_wav_input_shape):
            embed = unpack(embed, packed_wav_input_shape, '* t d')
            embed = torch.cat(embed, dim=1)

        if self.normalize_embeds:
            embed = zero_mean_unit_var_norm(embed)

        if return_embed:
            return embed

        embed, packed_shape = pack([embed], '* d')
        codebook_indices = self.kmeans.predict(embed.detach().cpu().numpy())
        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()

        if flatten:
            return codebook_indices

        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        return codebook_indices


def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def learn_kmeans(
    feat,
    seed,
    km_path='./results/kmeans.joblib',
    n_clusters=1024,
    init="k-means++",
    max_iter=100,
    batch_size=10000,
    tol=0.0,
    n_init=20,
    reassignment_ratio=0.0,
    max_no_improvement=100,
):
    np.random.seed(seed)
    km_model = get_kmeans_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    print("total intertia: %.5f", inertia)
    print("finished successfully")


def get_hubert_kmeans(model_name: str="m-a-p/MERT-v0", kmeans_path: Optional[str]='./checkpoints/kmeans.joblib', **kwargs):
    wav2vec = HubertModel.from_pretrained(model_name)
    kmeans = joblib.load(kmeans_path) if exists(kmeans_path) else None

    return HfHubertWithKmeans(hubert=wav2vec, kmeans=kmeans, **kwargs)
