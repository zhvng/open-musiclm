from audiolm_pytorch import HubertWithKmeans, SoundStream
from beartype.typing import Union

from .hf_hubert_kmeans import HfHubertWithKmeans
from .encodec_wrapper import EncodecWrapper

Wav2Vec = Union[HubertWithKmeans, HfHubertWithKmeans]
NeuralCodec = Union[SoundStream, EncodecWrapper]
