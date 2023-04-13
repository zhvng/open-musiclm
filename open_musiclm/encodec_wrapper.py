import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from encodec import EncodecModel
from torch import nn

from .utils import exists, beartype_jit


@beartype_jit
class EncodecWrapper(nn.Module):
    def __init__(self,
                 *,
                 encodec: EncodecModel,
                 output_hz: int = 75,
                 ):
        super().__init__()

        self.encodec = encodec
        self.sample_rate = encodec.sample_rate
        self.output_hz = output_hz

        assert exists(encodec.bandwidth)
        total_quantizers = encodec.quantizer.n_q
        self.num_quantizers = int(encodec.bandwidth / 24 * total_quantizers) # output quantizers per frame
        self.codebook_size = encodec.quantizer.bins

    def forward(self, x: torch.Tensor, return_encoded = True, **kwargs):
        assert return_encoded == True

        if x.dim() == 2:
            x = rearrange(x, 'b t -> b 1 t') # add in "mono" dimension

        with torch.no_grad():
            self.encodec.eval()
            encoded_frames = self.encodec.encode(x)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
        codes = rearrange(codes, 'b n_q t -> b t n_q')

        return None, codes, None # [B, T, n_q]

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

def create_encodec_24khz(bandwidth: float = 6.0, codebook_size: int = 1024, **kwargs):
    """
    Create a pretrained EnCodec model.
    Args:
        bandwidth: float, target bandwidth in kHz"""
    assert bandwidth in [1.5, 3., 6., 12., 24.], "invalid bandwidth. must be one of [1.5, 3., 6., 12., 24.]"

    encodec = EncodecModel.encodec_model_24khz()
    encodec.set_target_bandwidth(bandwidth)
    encodec_wrapper = EncodecWrapper(encodec=encodec, **kwargs)

    assert encodec_wrapper.codebook_size == codebook_size, "encodec codebook size must be 1024 for now"

    return encodec_wrapper