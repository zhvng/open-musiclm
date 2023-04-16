from pathlib import Path

import torch
from torch import nn
import numpy as np
from einops import rearrange, pack, unpack
from beartype.typing import Optional

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
    """

    def __init__(
        self,
        *,
        hubert: HubertModel,
        kmeans: Optional[MiniBatchKMeans] = None,
        embed_layer: int=7,
        target_sample_hz=16000,
        seq_len_multiple_of=int(16000 / 50),
        normalize_embeds=True,
        codebook_size: int=1024,
        output_hz: int=50
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.output_hz = output_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.codebook_size = kmeans.n_clusters if exists(kmeans) else None

        self.codebook_size = codebook_size
        if exists(kmeans):
            assert self.codebook_size == kmeans.n_clusters, "codebook_size must match kmeans.n_clusters"

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

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        hubert_args = {
            'input_values': wav_input,
            'attention_mask': torch.ones_like(wav_input, device=device), # TODO: handle padding
        }

        outputs = self.hubert(**hubert_args, output_hidden_states = True)
        embed = outputs.hidden_states[self.embed_layer]

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
