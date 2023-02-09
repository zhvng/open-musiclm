import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Dict, List, Literal, Optional, Union
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import nn
from transformers import RobertaTokenizer
from vector_quantize_pytorch import ResidualVQ

from .utils import exists


@beartype
class EncodecWrapper(nn.Module):
    def __init__(self,
                 *,
                 encodec: Optional[EncodecModel] = None
                 ):
        super().__init__()

        if not exists(encodec):
            encodec = EncodecModel.encodec_model_24khz()

        self.encodec = encodec

    def forward(self, x: torch.Tensor, return_encoded = True, **kwargs):
        assert return_encoded == True

        with torch.no_grad():
            self.encodec.eval()
            encoded_frames = self.encodec.encode(x)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        codes = rearrange(codes, 'b n_q t -> b t n_q')

        return codes # [B, T, n_q]

    def decode_from_codebook_indices(self, quantized_indices):
        """
        Args:
            quantized_indices: [B, T, n_q]
        """
        quantized_indices = rearrange(quantized_indices, 'b t n_q -> b n_q t')

        frames = [(quantized_indices, None)] # 1 frame for now
        with torch.no_grad():
            self.encodec.eval()
            wave = self.encodec.decode(frames)
        return wave
