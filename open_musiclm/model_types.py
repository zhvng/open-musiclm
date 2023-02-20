from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans, SoundStream
from open_musiclm.hf_hubert_kmeans import HfHubertWithKmeans
from beartype.typing import Union

from .encodec_wrapper import EncodecWrapper

Wav2Vec = Union[FairseqVQWav2Vec, HubertWithKmeans, HfHubertWithKmeans]
NeuralCodec = Union[SoundStream, EncodecWrapper]