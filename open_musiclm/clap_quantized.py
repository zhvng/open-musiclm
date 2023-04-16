import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from beartype.typing import Dict, List, Optional, Union
from einops import rearrange
from torch import nn
from transformers import RobertaTokenizer
from vector_quantize_pytorch import ResidualVQ

from .laion_clap import CLAP_Module
from .utils import exists, beartype_jit


@beartype_jit
class ClapQuantized(nn.Module):
    def __init__(self,
                 *,
                 clap: CLAP_Module,
                 codebook_size: int = 1024,
                 rq_num_quantizers: int = 12,
                 rq_ema_decay: float = 0.95,
                 learn_rvq: bool = False,
                 threshold_ema_dead_code: float = 0.0,
                 ):
        super().__init__()

        self.clap = clap
        self.codebook_size = codebook_size
        self.learn_rvq = learn_rvq

        self.sample_rate = self.clap.model_cfg['audio_cfg']['sample_rate']

        for param in self.clap.parameters():
            param.requires_grad = False

        self.rq = ResidualVQ(
            dim=clap.model.joint_embed_shape,
            num_quantizers=rq_num_quantizers,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
            commitment_weight=0,  # embeddings are frozen so no need for commitment loss
            decay=rq_ema_decay,
            kmeans_init=True,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

    def forward(self,
                *,
                audio_input: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
                text_input: Optional[List[str]] = None,
                return_embedding: Optional[bool] = False,
                return_rvq_loss = False,
                ):
        """
        Wrapper for clap module that takes in audio or text and returns the quantized embedding from the respective tower
        """

        assert exists(audio_input) ^ exists(text_input), "either audio or text must be provided, but not both"
        if exists(audio_input):
            assert all(wave.dim() == 1 for wave in audio_input), f"audio_input must be a list of 1D tensors, but got {audio_input[0].shape}"

        with torch.no_grad():
            self.clap.eval()
            if exists(audio_input):
                embedding = self.clap.get_audio_embedding_from_data(audio_input)
            else:
                embedding = self.clap.get_text_embedding(text_input)

        if return_embedding:
            return embedding

        return self.quantize(embedding, return_rvq_loss=return_rvq_loss)

    def quantize(self, embedding, return_rvq_loss=False):
        """
        Quantize an embedding and optionally return the loss
        """
        with torch.set_grad_enabled(self.learn_rvq):
            self.rq.train(self.learn_rvq)
            q, indices, _ = self.rq(rearrange(embedding, 'n c -> n 1 c'))

        if return_rvq_loss:
            return F.mse_loss(q, rearrange(embedding, 'n c -> n 1 c')).item()

        indices = rearrange(indices, 'n 1 c -> n c 1')
        return indices


def create_clap_quantized(
    device=None,
    learn_rvq=False,
    enable_fusion=False,
    rvq_checkpoint_path=None,
    checkpoint_path: Optional[str] = None,
    amodel_type: str = 'HTSAT-tiny',
    **kwargs
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clap = CLAP_Module(enable_fusion=enable_fusion, device=device, amodel=amodel_type)
    clap.load_ckpt(ckpt=checkpoint_path)

    clap_quantized = ClapQuantized(clap=clap, learn_rvq=learn_rvq, **kwargs)

    if exists(rvq_checkpoint_path):
        rvq = torch.load(rvq_checkpoint_path, map_location=device)
        clap_quantized.rq.load_state_dict(rvq)

    return clap_quantized
