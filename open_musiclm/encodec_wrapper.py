import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union, Literal
from clap import CLAP
from einops import rearrange
from torch import nn
from transformers import RobertaTokenizer
from utils import exists
from vector_quantize_pytorch import ResidualVQ
from encodec import EncodecModel
from encodec.utils import convert_audio


@beartype
class EncodecWrapper(nn.Module):
    def __init__(self,
                 *,
                 encodec: EncodecModel
                 ):
        super().__init__()

        self.encodec = encodec

    def encode(self, *, wav: torch.Tensor):

        with torch.no_grad():
            self.encodec.eval()
            encoded_frames = self.encodec.encode(wav)
        # codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

        return encoded_frames

    def decode(self, *, encoded_frames):
        with torch.no_grad():
            self.encodec.eval()
            wave = self.encodec.decode(encoded_frames)
        return wave
