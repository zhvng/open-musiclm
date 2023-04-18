import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from beartype import beartype
from pathlib import Path
import shutil
import os
from torchaudio.functional import resample

from einops import rearrange, repeat, reduce

def beartype_jit(func):
    """decorator to enable beartype only if USE_BEARTYPE is set to 1"""
    return beartype(func) if os.environ.get('USE_BEARTYPE', '0') == '1' else func

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

def remainder_needed_until_multiple(n, mult):
    return (ceil_div(n, mult) * mult) - n

def round_down_nearest_multiple(val, mult):
    return (val // mult) * mult

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

# attention related utils

def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def mask_out_after_eos_id(t, eos_id, mask_value = -1, keep_eos = True):
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim = -1) > 0
    return t.masked_fill(after_eos_mask, mask_value)

def all_rows_have_eos_id(t, eos_id):
    eos_mask = (t == eos_id)
    return torch.any(eos_mask, dim = -1).all()

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# removing unique consecutives in the semantic token ids
# important detail noted by @eonglints

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device = device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b = b)
    ids = torch.cat((ids, eos_ids), dim = -1)
    return ids

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

# to get embedding from sequence with padding token

@beartype_jit
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # just retrieve first code as dummy
    embeds = embeddings(codes_without_pad)

    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    if return_mask:
        return embeds, ~pad_mask

    return embeds

# audio processing helpers

def int16_to_float32(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def zero_mean_unit_var_norm(x):
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-7)

def prepare_audio(data, sample_hz, target_sample_hz, normalize=True, target_length_seconds=None):
    if data.shape[0] > 1:
        data = torch.mean(data, dim=0).unsqueeze(0)
    if normalize:
        data = zero_mean_unit_var_norm(data)
    if exists(target_length_seconds) and data.shape[1] > target_length_seconds * sample_hz:
        data = data[: , :int(target_length_seconds * sample_hz)]
    audio_for_wav2vec = resample(data, sample_hz, target_sample_hz)
    audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))
    return audio_for_wav2vec

# helper for saving config

def copy_file_to_folder(file_path: str, folder_path: str):
    config_file = Path(file_path)
    folder = Path(folder_path)

    shutil.copy(str(config_file), str(folder / config_file.name))