from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch


if __name__ == "__main__":

    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load("/u/zhvng/projects/audio_files/jumpman.mp3")
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    print(model.channels, model.sample_rate, model.quantizer.n_q)
    print(model.segment_stride)
    wav = torch.stack([wav, wav], dim=0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

    print(codes)
    print(codes.shape)
    print(encoded_frames[0][0].shape)
    print(len(encoded_frames))

    new_encoded_frames = [(encoded[0], None) for encoded in encoded_frames]

    wave = model.decode(new_encoded_frames)
    torchaudio.save('test.wav', wave[0], model.sample_rate)
