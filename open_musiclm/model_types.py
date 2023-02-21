from audiolm_pytorch import HubertWithKmeans, SoundStream
from open_musiclm.hf_hubert_kmeans import HfHubertWithKmeans
from beartype.typing import Union

from .encodec_wrapper import EncodecWrapper

Wav2Vec = Union[HubertWithKmeans, HfHubertWithKmeans]
NeuralCodec = Union[SoundStream, EncodecWrapper]