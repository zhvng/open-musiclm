"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""
import os
import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms
from contextlib import suppress
import numpy as np
from clap_module import create_model

from transformers import RobertaTokenizer
import wget
from clap_module.factory import load_state_dict


def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

class CLAP_Module(torch.nn.Module):
    def __init__(self, enable_fusion=False, device=None, amodel= 'HTSAT-tiny', tmodel='roberta') -> None:
        """Initialize CLAP Model

        Parameters
        ----------
        enable_fusion: bool
            if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false) 
        device: str
            if None, it will automatically detect the device (gpu or cpu)
        amodel: str
            audio encoder architecture, default: HTSAT-tiny
        tmodel: str
            text encoder architecture, default: roberta
        """
        super(CLAP_Module, self).__init__()
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        precision = 'fp32'

        if enable_fusion:
            fusion_type = 'aff_2d'
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type
            )
        else:
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion
            )
        self.enable_fusion = enable_fusion
        self.model = model
        self.model_cfg = model_cfg
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')

        audio_cfg = model_cfg['audio_cfg']
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
            n_mels=audio_cfg['mel_bins'],
            f_min=audio_cfg['fmin'],
            f_max=audio_cfg['fmax']
        )
        self.log_mel_transform = torchaudio.transforms.AmplitudeToDB(top_db=None)

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

    def load_ckpt(self, ckpt = None, model_id = -1):
        """Load the pretrained checkpoint of CLAP model

        Parameters
        ----------
        ckpt: str
            if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n 
            For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
        model_id:
            if model_id is specified, you can download our best ckpt, as:
                id = 0 --> 630k non-fusion ckpt \n
                id = 1 --> 630k+audioset non-fusion ckpt \n
                id = 2 --> 630k fusion ckpt \n
                id = 3 --> 630k+audioset fusion ckpt \n
            Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
        """
        download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
        download_names = [
            '630k-best.pt',
            '630k-audioset-best.pt',
            '630k-fusion-best.pt',
            '630k-audioset-fusion-best.pt'
        ]
        if ckpt is not None:
            print(f'Load the specified checkpoint {ckpt} from users.')
        else:
            print(f'Load our best checkpoint in the paper.')
            if model_id == -1:
                model_id = 3 if self.enable_fusion else 1
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weight_file_name = download_names[model_id]
            ckpt = os.path.join(package_dir, weight_file_name)
            if os.path.exists(ckpt):
                print(f'The checkpoint is already downloaded')
            else:
                print('Downloading laion_clap weight files...')
                ckpt = wget.download(download_link + weight_file_name, os.path.dirname(ckpt))
                print('Download completed!')
        print('Load Checkpoint...')
        ckpt = load_state_dict(ckpt, skip_params=True)
        self.model.load_state_dict(ckpt, strict=False)
        param_names = [n for n, p in self.model.named_parameters()]
        for n in param_names:
            print(n, "\t", "Loaded" if n in ckpt else "Unloaded")

    def get_mel(self, audio_data):
        mel = self.mel_transform(audio_data)
        mel = self.log_mel_transform(mel)
        return mel.T  # (T, n_mels)

    def get_audio_features(self, sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
        """
        Calculate and add audio features to sample.
        Sample: a dict containing all the data of current sample.
        audio_data: a tensor of shape (T) containing audio data.
        max_len: the maximum length of audio data.
        data_truncating: the method of truncating data.
        data_filling: the method of filling data.
        audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
        require_grad: whether to require gradient for audio data.
            This is useful when we want to apply gradient-based classifier-guidance.
        """
        grad_fn = suppress if require_grad else torch.no_grad
        with grad_fn():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    # fusion
                    mel = self.get_mel(audio_data)
                    # split to three parts
                    chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        # there is a corner case where the audio length is
                        # larger than max_len but smaller than max_len+hop_size.
                        # In this case, we just use the whole audio.
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
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
                        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                        # shrink the mel
                        mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]
                        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                        # stack
                        mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
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
                        n_repeat = int(max_len / len(audio_data))
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
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
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

    def get_audio_embedding_from_data(self, x):
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: torch.Tensor (N,T):
            audio data, must be mono audio tracks.      
        Returns
        ----------
        audio embed: torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        self.model.eval()
        audio_input = []
        for audio_waveform in x:          
            # quantize
            audio_waveform = int16_to_float32_torch(float32_to_int16_torch(audio_waveform))
            temp_dict = {}
            # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            temp_dict = self.get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        return audio_embed

    def get_text_embedding(self, x, tokenizer = None):
        """get text embeddings from texts

        Parameters
        ----------
        x: List[str] (N,): 
            text list 
        tokenizer: func:
            the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
        
        Returns
        ----------
        text_embed : torch.Tensor (N,D):
            text embeddings that extracted from texts
        """ 
        self.model.eval()
        if tokenizer is not None:
            text_input = tokenizer(x)
        else:
            text_input = self.tokenizer(x)
        text_embed = self.model.get_text_embedding(text_input)
        text_embed = text_embed
        return text_embed
        
    
