from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans, SoundStream
from beartype.typing import Union

from .encodec_wrapper import EncodecWrapper

Wav2Vec = Union[FairseqVQWav2Vec, HubertWithKmeans]
NeuralCodec = Union[SoundStream, EncodecWrapper]