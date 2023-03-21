from beartype.typing import Union

from .hf_hubert_kmeans import HfHubertWithKmeans
from .encodec_wrapper import EncodecWrapper

Wav2Vec = HfHubertWithKmeans
NeuralCodec = EncodecWrapper
