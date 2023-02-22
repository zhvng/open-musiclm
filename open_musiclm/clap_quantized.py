import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union
from einops import rearrange
from torch import nn
from transformers import RobertaTokenizer
from vector_quantize_pytorch import ResidualVQ

from .clap import CLAP, create_model
from .utils import exists


@beartype
class ClapQuantized(nn.Module):
    def __init__(self,
                 *,
                 clap: CLAP,
                 clap_cfg: Dict[str, any],
                 tokenizer: Optional[RobertaTokenizer] = None,
                 codebook_size: int = 1024,
                 rq_num_quantizers: int = 12,
                 rq_ema_decay: float = 0.95,
                 learn_rvq: bool = False,
                 ):
        super().__init__()

        self.clap = clap
        self.clap_cfg = clap_cfg
        self.codebook_size = codebook_size
        self.learn_rvq = learn_rvq

        audio_cfg = clap_cfg['audio_cfg']
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg['sample_rate'],
            n_fft=audio_cfg['window_size'],
            win_length=audio_cfg['window_size'],
            hop_length=audio_cfg['hop_size'],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=audio_cfg['fmin'],
            f_max=audio_cfg['fmax']
        )
        self.log_mel_transform = torchaudio.transforms.AmplitudeToDB(top_db=None)

        self.sample_rate = audio_cfg['sample_rate']

        if not exists(tokenizer):
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer = tokenizer

        for param in self.clap.parameters():
            param.requires_grad = False

        self.rq = ResidualVQ(
            dim=clap.joint_embed_shape,
            num_quantizers=rq_num_quantizers,  # specify number of quantizers
            codebook_size=codebook_size,  # codebook size
            commitment_weight=0,  # embeddings are frozen so no need for commitment loss
            decay=rq_ema_decay,
            kmeans_init=True,
            threshold_ema_dead_code=1,
        )

    def get_mel(self, audio_data):
        mel = self.mel_transform(audio_data)
        mel = self.log_mel_transform(mel)
        return mel.T  # (T, n_mels)
        

    def get_audio_features(self, sample, audio_data, max_len, data_truncating, data_filling):
        """
        Calculate audio features. Code from CLAP (github.com/LAION/CLAP)
        For fusion features, we split audio above max_len into parts of length max_len and stack them. 
            If the audio is shorter than max_len, we first pad it with zeros and then stack.
        """
        audio_cfg = self.clap_cfg['audio_cfg']
        with torch.no_grad():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    # fusion
                    mel = self.get_mel(audio_data)
                    # split to three parts
                    # the +1 related to how the spectrogram is computed
                    chunk_frames = max_len // audio_cfg['hop_size']+1
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        # there is a corner case where the audio length is
                        # larger than max_len but smaller than max_len+hop_size.
                        # In this case, we just use the whole audio.
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(
                            list(range(0, total_frames-chunk_frames+1)), 3)
                        # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                        #       'len(audio_data):', len(audio_data),
                        #       'chunk_frames:', chunk_frames,
                        #       'total_frames:', total_frames)
                        if len(ranges[1]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[1] = [0]
                        if len(ranges[2]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[2] = [0]
                        # randomly choose index for each part
                        idx_front = np.random.choice(ranges[0])
                        idx_middle = np.random.choice(ranges[1])
                        idx_back = np.random.choice(ranges[2])
                        # select mel
                        mel_chunk_front = mel[idx_front:idx_front+chunk_frames, :]
                        mel_chunk_middle = mel[idx_middle:idx_middle +
                                            chunk_frames, :]
                        mel_chunk_back = mel[idx_back:idx_back+chunk_frames, :]

                        # shrink the mel
                        mel_shrink = torchvision.transforms.Resize(
                            size=[chunk_frames, 64])(mel[None])[0]
                        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                        # stack
                        mel_fusion = torch.stack(
                            [mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([True])
                else:
                    raise NotImplementedError(
                        f"data_truncating {data_truncating} not implemented"
                    )
                # random crop to max_len (for compatibility)
                overflow = len(audio_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                audio_data = audio_data[idx: idx + max_len]

            else:  # padding if too short
                if len(audio_data) < max_len:  # do nothing if equal
                    if data_filling == "repeatpad":
                        n_repeat = int(max_len/len(audio_data))
                        audio_data = audio_data.repeat(n_repeat)
                        # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "pad":
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "repeat":
                        n_repeat = int(max_len/len(audio_data))
                        audio_data = audio_data.repeat(n_repeat+1)[:max_len]
                    else:
                        raise NotImplementedError(
                            f"data_filling {data_filling} not implemented"
                        )
                if data_truncating == 'fusion':
                    mel = self.get_mel(audio_data)
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                longer = torch.tensor([False])

        sample["longer"] = longer
        sample["waveform"] = audio_data

        return sample

    def tokenize(self, text):
        result = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

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
            assert all(wave.dim() == 1 for wave in audio_input)

        with torch.no_grad():
            self.clap.eval()
            if exists(audio_input):
                audio_dicts = []
                for waveform in audio_input:
                    audio_dict = self.get_audio_features({}, waveform, 480000, data_truncating='fusion', data_filling='repeatpad')
                    audio_dicts.append(audio_dict)

                embedding = self.clap.get_audio_embedding(audio_dicts)
            else:
                text_input = self.tokenize(text_input)
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


def create_clap_quantized(device=None, learn_rvq=False, checkpoint_path="./checkpoints/clap-laion-audioset-fusion.pt", rvq_checkpoint_path=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    precision = 'fp32'
    amodel = 'HTSAT-tiny'  # or 'PANN-14'
    tmodel = 'roberta'  # the best text encoder in our training
    enable_fusion = True  # False if you do not want to use the fusion model
    fusion_type = 'aff_2d'
    # the checkpoint name, the unfusion model can also be loaded.
    pretrained = checkpoint_path

    model, model_cfg = create_model(
        amodel,
        tmodel,
        pretrained,
        precision=precision,
        device=device,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
    )

    clap = ClapQuantized(clap=model, clap_cfg=model_cfg, learn_rvq=learn_rvq)

    if exists(rvq_checkpoint_path):
        rvq = torch.load(rvq_checkpoint_path, map_location=device)
        clap.rq.load_state_dict(rvq)

    return clap
