import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm
from beartype import beartype
from beartype.typing import List, Optional
from einops import rearrange, reduce, repeat
from torch import einsum, nn

from .clap_quantized import ClapQuantized
from .model_types import NeuralCodec, Wav2Vec
from .transformer import Transformer
from .utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, ceil_div, default,
                    eval_decorator, exists, generate_mask_with_prob,
                    get_embeds, gumbel_sample, mask_out_after_eos_id,
                    round_down_nearest_multiple, top_k)


@dataclass
class TokenSequenceInfo():
    """
    Defines a token sequence to be conditioned on or predicted in TokenConditionedTransformer
    """
    codebook_size: int
    num_quantizers: int    # e.g. 1 for semantic, Q for coarse acoustic, ...
    unique_consecutive: bool    # whether to remove unique consecutive tokens. see https://github.com/lucidrains/audiolm-pytorch/discussions/13#discussioncomment-4117105


class TokenConditionedTransformer(nn.Module):
    """
    Combination of the SemanticTransformer, CoarseTransformer and FineTransformer in lucidrain's AudioLM implementation.
    Can handle a variable number of token sequences, each with their own parameters. 
    https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/audiolm_pytorch.py
    """
    # TODO: Add in text conditioning for parity with AudioLM. Not important for MusicLM though.

    def __init__(
        self,
        *,
        token_sequences: List[TokenSequenceInfo],
        dim,
        depth,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.1,
        has_condition=False,
        cond_as_self_attn_prefix=False,
        cond_drop_prob=0.5,
        grad_shrink_alpha=0.1,
        **kwargs
    ):
        super().__init__()

        self.token_sequences = token_sequences

        self.has_condition = has_condition
        self.cond_drop_prob = cond_drop_prob

        self.start_tokens = torch.nn.ParameterList()
        self.logit_weights = torch.nn.ParameterList()
        self.embeddings = torch.nn.ModuleList()
        self.eos_ids = []
        
        for sequence in token_sequences:
            self.start_tokens.append(nn.Parameter(torch.randn(dim)))
            self.eos_ids.append(sequence.codebook_size)

            codebook_size_with_eos = sequence.codebook_size + 1

            self.embeddings.append(nn.Embedding(codebook_size_with_eos * sequence.num_quantizers, dim))
            self.logit_weights.append(nn.Parameter(torch.randn(sequence.num_quantizers, codebook_size_with_eos, dim)))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            cross_attend=has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix=cond_as_self_attn_prefix,
            grad_shrink_alpha=grad_shrink_alpha,
            **kwargs
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,
                *,
                all_token_ids: List[torch.Tensor],
                self_attn_mask=None,
                cond_drop_prob=None,
                return_only_final_seq_logits=False
                ):
        """
        all_token_ids: List of tensors containing token ids. Each element can either be 2 dimensional (batch_size, n_time_steps * num_quantizers) or 3 dimensional (batch_size, n_time_steps, num_quantizers)
                       Each element in list corresponds to one token sequence in self.token_sequences (e.g. semantic, coarse acoustic, fine acoustic, etc.)

        return_only_final_seq_logits: If True, only return logits for the final token sequence in self.token_sequences.
        """

        b, device = all_token_ids[0].shape[0], self.device

        all_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids))

        assert len(all_token_ids) == len(self.token_sequences) == len(self.embeddings)

        tokens = []
        start_tokens = []
        split_at = []
        for sequence, token_ids, embedding, start_token in zip(self.token_sequences, all_token_ids, self.embeddings, self.start_tokens):

            # add offsets
            if sequence.num_quantizers > 1:
                offsets = sequence.codebook_size * torch.arange(sequence.num_quantizers, device=device)
                offsets = repeat(offsets, 'q -> 1 (n q)', n=ceil_div(token_ids.shape[-1], sequence.num_quantizers))
                offsets = offsets[:, :token_ids.shape[-1]]
                token_ids = token_ids + offsets

            # get embeddings and prepare for next step
            token_embeddings = get_embeds(embedding, token_ids, pad_id=-1) if sequence.unique_consecutive else embedding(token_ids)
            tokens.append(token_embeddings)
            start_tokens.append(repeat(start_token, 'd -> b 1 d', b=b))

            n_tokens = token_embeddings.shape[1] + 1 # +1 for end token
            split_at.append(n_tokens if len(split_at) == 0 else split_at[-1] + n_tokens)

        tokens = list(itertools.chain(*zip(start_tokens, tokens)))  # [start_1, tokens_1, start_2, tokens_2, ...]
        tokens = torch.cat(tokens, dim=1)

        tokens = self.transformer(tokens, self_attn_mask=self_attn_mask)

        split_at = split_at[:-1]  # remove last element (total number of tokens)
        all_pred_tokens = torch.tensor_split(tokens, split_at, dim=1)

        # strip eos token from all sequences besides end
        all_pred_tokens = [pred_tokens[:, :-1] for pred_tokens in all_pred_tokens[:-1]] +  [all_pred_tokens[-1]]

        # get logits

        all_logits = []
        assert len(all_pred_tokens) == len(self.token_sequences) == len(self.logit_weights)

        for index, (sequence, pred_tokens, seq_logit_weights) in enumerate(zip(self.token_sequences, all_pred_tokens, self.logit_weights)):
            if not return_only_final_seq_logits or index == len(self.token_sequences) - 1:
                n = pred_tokens.shape[1]
                nq = round_down_nearest_multiple(n, sequence.num_quantizers)

                pred_tokens_groupable, pred_tokens_remainder = pred_tokens[:, :nq], pred_tokens[:, nq:]

                pred_tokens_groupable = rearrange(
                    pred_tokens_groupable, 'b (n q) d -> b n q d', q=sequence.num_quantizers)

                pred_logits_groupable = einsum('q c d, b n q d -> b n q c', seq_logit_weights, pred_tokens_groupable)

                pred_logits_groupable = rearrange(pred_logits_groupable, 'b n q c -> b (n q) c')

                remainder_num_tokens_in_step = pred_tokens_remainder.shape[1]

                if remainder_num_tokens_in_step > 0:
                    pred_logits_remainder = einsum(
                        'q c d, b q d -> b q c', seq_logit_weights[:remainder_num_tokens_in_step], pred_tokens_remainder)
                    pred_logits = torch.cat((pred_logits_groupable, pred_logits_remainder), dim=1)
                else:
                    pred_logits = pred_logits_groupable

                all_logits.append(pred_logits)
            else:
                all_logits.append(None)

        return all_logits

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=3,
        **kwargs
    ):
        """Doesn't do anything without the AudioLM-pytorch text conditioning implementation. Just use forward() instead."""

        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        scaled_logits = []

        for seq_logits, null_seq_logits in zip(logits, null_logits):
            if seq_logits is None:
                scaled_logits.append(None)
            else:
                scaled_logits.append(null_seq_logits + (seq_logits - null_seq_logits) * cond_scale)

        return scaled_logits


@beartype
class TokenConditionedTransformerWrapper(nn.Module):
    """Combination of SemanticTransformerWrapper, CoarseTransformerWrapper and FineTransformerWrapper in lucidrain's audiolm-pytorch, without the input processing + text conditioning"""
    def __init__(
        self,
        *,
        transformer: TokenConditionedTransformer,
        pad_id=-1,
        unique_consecutive=True,
        cross_entropy_loss_weights: Optional[List[float]] = None,
        mask_prob=0.15
    ):
        super().__init__()

        self.transformer = transformer

        self.token_sequences = transformer.token_sequences

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.cross_entropy_loss_weights = default(cross_entropy_loss_weights, [1 for _ in self.token_sequences])

        self.eos_ids = transformer.eos_ids
        self.mask_prob = mask_prob

        assert len(self.token_sequences) == len(self.eos_ids) == len(self.cross_entropy_loss_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        conditioning_token_ids: List[torch.Tensor],
        pred_token_ids: Optional[torch.Tensor] = None,
        max_time_steps=512,
        filter_thres=0.9,
        temperature=1.,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        **kwargs
    ):
        assert len(conditioning_token_ids) == len(self.token_sequences) - 1

        batch, device = conditioning_token_ids[0].shape[0], self.device

        conditioning_token_ids = [t.to(device) for t in conditioning_token_ids]

        if exists(pred_token_ids):
            assert pred_token_ids.shape[0] == batch
            init_pred_time_step = pred_token_ids.shape[1]
            pred_token_ids = rearrange(pred_token_ids, 'b ... -> b (...)')
        else:
            init_pred_time_step = 0
            pred_token_ids = torch.empty((batch, 0), device=device, dtype=torch.long)

        pred_sequence_info, pred_eos_id = self.token_sequences[-1], self.eos_ids[-1]

        # batch unique consecutive
        for index, sequence_info in enumerate(self.token_sequences[:-1]):
            if sequence_info.unique_consecutive:
                conditioning_token_ids[index] = batch_unique_consecutive(
                    conditioning_token_ids[index], pad_value=self.pad_id)
        if self.token_sequences[-1].unique_consecutive:
            pred_token_ids = batch_unique_consecutive(pred_token_ids, pad_value=self.pad_id)

        # reshape and append eos
        if append_eos_to_conditioning_tokens:
            conditioning_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), conditioning_token_ids))
            conditioning_token_ids = [append_eos_id(ids, eos_id) for ids, eos_id in zip(conditioning_token_ids, self.eos_ids)]

        # initialize

        sampled_pred_token_ids = pred_token_ids.clone()

        for time_step in tqdm(range(init_pred_time_step, max_time_steps), desc='generating predicted tokens'):
            for ind in range(pred_sequence_info.num_quantizers):
                is_last_step = ind == (pred_sequence_info.num_quantizers - 1)

                pred_logits = self.transformer(
                    all_token_ids=conditioning_token_ids + [sampled_pred_token_ids],
                    return_only_final_seq_logits=True,
                    **kwargs
                )[-1]

                last_pred_logits = pred_logits[:, -1]

                if not is_last_step:
                    # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval
                    last_pred_logits[:, -1] = float('-inf')

                filtered_logits = top_k(last_pred_logits, thres=filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_pred_token_ids = torch.cat((sampled_pred_token_ids, sampled), dim=-1)

        sampled_pred_token_ids = mask_out_after_eos_id(
            sampled_pred_token_ids, pred_eos_id, keep_eos=include_eos_in_output)
        sampled_pred_token_ids = rearrange(
            sampled_pred_token_ids, 'b (n q) -> b n q', q=pred_sequence_info.num_quantizers)

        return sampled_pred_token_ids

    def forward(
        self,
        *,
        all_token_ids: List[torch.Tensor],
        return_loss: bool=False,
        input_has_eos: bool=False,
        **kwargs
    ):
        assert len(all_token_ids) == len(self.token_sequences)

        batch, device = all_token_ids[0].shape[0], self.device

        all_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids))

        if self.training:
            assert not input_has_eos, "train sequences (from clap, wav2vec, etc.) shouldn't come with an eos token"
        
        # append eos to sequences if not already there
        if not input_has_eos:
            all_token_ids = [append_eos_id(ids, eos_id) for ids, eos_id in zip(all_token_ids, self.eos_ids)]

        if self.unique_consecutive:
            for index, sequence_info in enumerate(self.token_sequences):
                if sequence_info.unique_consecutive:
                    all_token_ids[index] = batch_unique_consecutive(all_token_ids[index], pad_value=self.pad_id)

        if return_loss:
            all_labels = [ids.clone() for ids in all_token_ids]
            all_token_ids[-1] = all_token_ids[-1][:, :-1]  # don't include last token when returning loss (should be eos)

        # do not attend to padding tokens or eos tokens
        combined_self_attn_mask = torch.empty((batch, 0), device=device, dtype=torch.bool)
        for ids, eos_id in zip(all_token_ids[:-1], self.eos_ids[:-1]):
            mask = (ids != self.pad_id) & (ids != eos_id)

            ids.masked_fill_(~mask, 0)  # inplace

            # transformer appends a start token to beginning of sequence, so add to mask
            mask = F.pad(mask, (1, 0), value=True)
            combined_self_attn_mask = torch.cat((combined_self_attn_mask, mask), dim=-1)

        # add our predicted tokens + start token to our mask
        pred_token_len = all_token_ids[-1].shape[-1]
        combined_self_attn_mask = F.pad(combined_self_attn_mask, (0, pred_token_len + 1), value=True)

        # forgetful causal mask - structured dropout
        if self.mask_prob > 0 and self.training:
            combined_self_attn_mask &= generate_mask_with_prob(
                combined_self_attn_mask.shape, self.mask_prob, device=combined_self_attn_mask.device)

        all_logits = self.transformer(
            all_token_ids=all_token_ids,
            self_attn_mask=combined_self_attn_mask,
            **kwargs
        )

        # whether to early return the logits

        if not return_loss:
            return all_logits

        all_logits = list(map(lambda t: rearrange(t, 'b n c -> b c n'), all_logits))

        total_logits = 0
        running_loss = 0.
        for logits, labels, cross_entropy_loss_weight, sequence_info in zip(all_logits, all_labels, self.cross_entropy_loss_weights, self.token_sequences):
            loss = 0.
            num_logits = 0
            unique_consecutive = sequence_info.unique_consecutive and self.unique_consecutive

            if cross_entropy_loss_weight > 0 and exists(logits):
                num_logits = (labels != self.pad_id).sum() if unique_consecutive else labels.numel()

                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.pad_id if unique_consecutive else -100 
                )

            total_logits += num_logits
            running_loss += loss * num_logits * cross_entropy_loss_weight

        return running_loss / total_logits


@beartype
def create_semantic_transformer(
    dim=1024,
    depth=6,
    clap_codebook_size=1024,
    semantic_codebook_size=1024,
    num_clap_quantizers=12,
    **kwargs
):

    clap_sequence = TokenSequenceInfo(codebook_size=clap_codebook_size, num_quantizers=num_clap_quantizers,
                                      unique_consecutive=False)
    semantic_sequence = TokenSequenceInfo(codebook_size=semantic_codebook_size,
                                          num_quantizers=1, unique_consecutive=True)

    return TokenConditionedTransformer(token_sequences=[clap_sequence, semantic_sequence], dim=dim, depth=depth, **kwargs)


@beartype
def create_coarse_transformer(
    dim=512,
    depth=6,
    clap_codebook_size=1024,
    semantic_codebook_size=1024,
    acoustic_codebook_size=1024,
    num_clap_quantizers=12,
    num_coarse_quantizers=4,
    **kwargs
):

    clap_sequence = TokenSequenceInfo(codebook_size=clap_codebook_size, num_quantizers=num_clap_quantizers,
                                      unique_consecutive=False)
    semantic_sequence = TokenSequenceInfo(codebook_size=semantic_codebook_size,
                                          num_quantizers=1, unique_consecutive=True)
    coarse_sequence = TokenSequenceInfo(
        codebook_size=acoustic_codebook_size, num_quantizers=num_coarse_quantizers, unique_consecutive=False)

    return TokenConditionedTransformer(token_sequences=[clap_sequence, semantic_sequence, coarse_sequence], dim=dim, depth=depth, **kwargs)


@beartype
def create_fine_transformer(
    dim=512,
    depth=6,
    clap_codebook_size=1024,
    acoustic_codebook_size=1024,
    num_clap_quantizers=12,
    num_coarse_quantizers=4,
    num_fine_quantizers=8,
    **kwargs
):

    clap_sequence = TokenSequenceInfo(codebook_size=clap_codebook_size, num_quantizers=num_clap_quantizers,
                                      unique_consecutive=False)
    coarse_sequence = TokenSequenceInfo(codebook_size=acoustic_codebook_size,
                                        num_quantizers=num_coarse_quantizers, unique_consecutive=False)
    fine_sequence = TokenSequenceInfo(
        codebook_size=acoustic_codebook_size, num_quantizers=num_fine_quantizers, unique_consecutive=False)

    return TokenConditionedTransformer(token_sequences=[clap_sequence, coarse_sequence, fine_sequence], dim=dim, depth=depth, **kwargs)


@beartype
def get_or_compute_clap_token_ids(clap_token_ids: Optional[torch.Tensor], clap: Optional[ClapQuantized], conditioning_audio: Optional[torch.Tensor], conditioning_text: Optional[List[str]]):
    if not exists(clap_token_ids):
        assert exists(conditioning_audio) ^ exists(conditioning_text), "either condition on text or audio"
        assert exists(clap)
        if exists(conditioning_text):
            clap_token_ids = clap(text_input=conditioning_text)
        else:
            clap_token_ids = clap(audio_input=conditioning_audio)

    return clap_token_ids


@beartype
def get_or_compute_semantic_token_ids(semantic_token_ids: Optional[torch.Tensor], raw_audio: Optional[torch.Tensor], wav2vec: Optional[Wav2Vec]):
    if not exists(semantic_token_ids):
        assert exists(raw_audio)
        assert exists(wav2vec)
        semantic_token_ids = wav2vec(raw_audio, flatten=False)

    return semantic_token_ids


@beartype
def get_or_compute_acoustic_token_ids(coarse_token_ids: Optional[torch.Tensor], fine_token_ids: Optional[torch.Tensor], raw_audio: Optional[torch.Tensor], neural_codec: Optional[NeuralCodec], num_coarse_quantizers: int):

    if exists(raw_audio):
        assert not exists(coarse_token_ids) and not exists(fine_token_ids), "either provide coarse + fine ids or raw audio"
        assert exists(neural_codec), 'A neural audio codec must be provided if given raw wave for training'

        with torch.no_grad():
            neural_codec.eval()
            _, indices, _ = neural_codec(raw_audio, return_encoded=True)
            coarse_token_ids, fine_token_ids = indices[..., :num_coarse_quantizers], indices[..., num_coarse_quantizers:]

    return coarse_token_ids, fine_token_ids


@beartype
class SemanticStage(nn.Module):
    def __init__(
        self,
        *,
        semantic_transformer: TokenConditionedTransformer,
        wav2vec: Optional[Wav2Vec] = None,
        clap: Optional[ClapQuantized] = None,
        pad_id=-1,
        unique_consecutive=True,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15
    ):
        super().__init__()

        self.wav2vec = wav2vec
        self.clap = clap

        num_semantic_tokens = semantic_transformer.token_sequences[1].codebook_size

        if exists(wav2vec):
            assert self.wav2vec.codebook_size == num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.transformer_wrapper = TokenConditionedTransformerWrapper(
            transformer=semantic_transformer,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights,
            mask_prob=mask_prob
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        conditioning_text: Optional[List[str]] = None,
        conditioning_audio: Optional[torch.Tensor] = None,
        input_audio: Optional[torch.Tensor] = None,
        clap_token_ids: Optional[torch.Tensor] = None,
        semantic_token_ids: Optional[torch.Tensor] = None,
        filter_thres=0.9,
        temperature=1.,
        max_time_steps=30*25,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        **kwargs
    ):

        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids, self.clap, conditioning_audio, conditioning_text)

        semantic_token_ids = get_or_compute_semantic_token_ids(semantic_token_ids, input_audio, self.wav2vec) if (
            exists(semantic_token_ids) or exists(input_audio)) else None

        sampled_tokens = self.transformer_wrapper.generate(
            conditioning_token_ids=[clap_token_ids],
            pred_token_ids=semantic_token_ids,
            max_time_steps=max_time_steps,
            filter_thres=filter_thres,
            temperature=temperature,
            include_eos_in_output=include_eos_in_output,
            append_eos_to_conditioning_tokens=append_eos_to_conditioning_tokens,
            **kwargs
        )

        return sampled_tokens

    def forward(
        self,
        *,
        raw_wave_for_clap=None,
        raw_wave_for_semantic=None,
        clap_token_ids: Optional[torch.Tensor] = None,
        semantic_token_ids: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs
    ):
        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids, self.clap, raw_wave_for_clap, conditioning_text=None)
        semantic_token_ids = get_or_compute_semantic_token_ids(semantic_token_ids=semantic_token_ids, raw_audio=raw_wave_for_semantic, wav2vec=self.wav2vec)

        return self.transformer_wrapper.forward(
            all_token_ids=[clap_token_ids, semantic_token_ids],
            return_loss=return_loss,
            **kwargs
        )


@beartype
class CoarseStage(nn.Module):
    def __init__(
        self,
        *,
        coarse_transformer: TokenConditionedTransformer,
        wav2vec: Optional[Wav2Vec] = None,
        clap: Optional[ClapQuantized] = None,
        neural_codec: Optional[NeuralCodec] = None,
        pad_id=-1,
        unique_consecutive=True,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15
    ):
        super().__init__()

        self.wav2vec = wav2vec
        self.clap = clap
        self.neural_codec = neural_codec

        num_semantic_tokens = coarse_transformer.token_sequences[1].codebook_size

        if exists(wav2vec):
            assert self.wav2vec.codebook_size == num_semantic_tokens, f'num_semantic_tokens on CoarseTransformer must be set to {self.wav2vec.codebook_size}'

        self.num_coarse_quantizers = coarse_transformer.token_sequences[-1].num_quantizers

        self.transformer_wrapper = TokenConditionedTransformerWrapper(
            transformer=coarse_transformer,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights,
            mask_prob=mask_prob
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids: torch.Tensor,
        coarse_token_ids: Optional[torch.Tensor] = None,
        conditioning_text: Optional[List[str]] = None,
        conditioning_audio: Optional[torch.Tensor] = None,
        clap_token_ids: Optional[torch.Tensor] = None,
        filter_thres=0.9,
        temperature=1.,
        max_time_steps=10*600,
        include_eos_in_output=False,  # if doing hierarchical sampling, eos can be kept for an easy time
        append_eos_to_conditioning_tokens=True, # if doing heirarchical sampling and you want more control
        reconstruct_wave = False,
        **kwargs
    ):
        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids, self.clap, conditioning_audio, conditioning_text)

        sampled_tokens = self.transformer_wrapper.generate(
            conditioning_token_ids=[clap_token_ids, semantic_token_ids],
            pred_token_ids=coarse_token_ids,
            max_time_steps=max_time_steps,
            filter_thres=filter_thres,
            temperature=temperature,
            include_eos_in_output=include_eos_in_output,
            append_eos_to_conditioning_tokens=append_eos_to_conditioning_tokens,
            **kwargs
        )

        if reconstruct_wave:
            assert exists(self.neural_codec)
            wave = self.neural_codec.decode_from_codebook_indices(sampled_tokens)
            return rearrange(wave, 'b 1 n -> b n')
        
        return sampled_tokens

    def forward(
        self,
        *,
        raw_wave_for_clap=None,
        raw_wave_for_semantic=None,
        raw_wave_for_acoustic=None,
        clap_token_ids: Optional[torch.Tensor] = None,
        semantic_token_ids: Optional[torch.Tensor] = None,
        coarse_token_ids: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs
    ):
        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids=clap_token_ids, clap=self.clap,
                                                       conditioning_audio=raw_wave_for_clap, conditioning_text=None)
        semantic_token_ids = get_or_compute_semantic_token_ids(
            semantic_token_ids=semantic_token_ids, raw_audio=raw_wave_for_semantic, wav2vec=self.wav2vec)

        coarse_token_ids, _ = get_or_compute_acoustic_token_ids(
            coarse_token_ids=coarse_token_ids,
            fine_token_ids=None,
            raw_audio=raw_wave_for_acoustic,
            neural_codec=self.neural_codec,
            num_coarse_quantizers=self.num_coarse_quantizers
        )

        return self.transformer_wrapper.forward(
            all_token_ids=[clap_token_ids, semantic_token_ids, coarse_token_ids],
            return_loss=return_loss,
            **kwargs
        )


@beartype
class FineStage(nn.Module):
    def __init__(
        self,
        *,
        fine_transformer: TokenConditionedTransformer,
        clap: Optional[ClapQuantized] = None,
        neural_codec: Optional[NeuralCodec] = None,
        pad_id=-1,
        unique_consecutive=True,
        cross_entropy_loss_weights: List[float] = None,
        mask_prob=0.15
    ):
        super().__init__()

        self.clap = clap
        self.neural_codec = neural_codec

        self.num_coarse_quantizers = fine_transformer.token_sequences[1].num_quantizers

        self.transformer_wrapper = TokenConditionedTransformerWrapper(
            transformer=fine_transformer,
            pad_id=pad_id,
            unique_consecutive=unique_consecutive,
            cross_entropy_loss_weights=cross_entropy_loss_weights,
            mask_prob=mask_prob
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        coarse_token_ids: torch.Tensor,
        conditioning_text: Optional[List[str]] = None,
        conditioning_audio: Optional[torch.Tensor] = None,
        clap_token_ids: Optional[torch.Tensor] = None,
        filter_thres=0.9,
        temperature=1.,
        max_time_steps=3*600,
        include_eos_in_output=False,  # if doing hierarchical sampling, eos can be kept for an easy time
        append_eos_to_conditioning_tokens=True, # if doing heirarchical sampling and you want more control
        reconstruct_wave=False,
        **kwargs
    ):
        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids, self.clap, conditioning_audio, conditioning_text)
        fine_token_ids = None

        sampled_tokens = self.transformer_wrapper.generate(
            conditioning_token_ids=[clap_token_ids, coarse_token_ids],
            pred_token_ids=fine_token_ids,
            max_time_steps=max_time_steps,
            filter_thres=filter_thres,
            temperature=temperature,
            include_eos_in_output=include_eos_in_output,
            append_eos_to_conditioning_tokens=append_eos_to_conditioning_tokens,
            **kwargs
        )

        if reconstruct_wave:
            assert exists(self.neural_codec)
            coarse_and_fine_ids = torch.cat((coarse_token_ids, sampled_tokens), dim = -1)
            wave = self.neural_codec.decode_from_codebook_indices(coarse_and_fine_ids)
            return rearrange(wave, 'b 1 n -> b n')
        
        return sampled_tokens

    def forward(
        self,
        *,
        raw_wave_for_clap=None,
        raw_wave_for_acoustic=None,
        clap_token_ids: Optional[torch.Tensor] = None,
        coarse_token_ids: Optional[torch.Tensor] = None,
        fine_token_ids: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs
    ):
        clap_token_ids = get_or_compute_clap_token_ids(clap_token_ids=clap_token_ids, clap=self.clap,
                                                       conditioning_audio=raw_wave_for_clap, conditioning_text=None)

        coarse_token_ids, fine_token_ids = get_or_compute_acoustic_token_ids(
            coarse_token_ids=coarse_token_ids,
            fine_token_ids=fine_token_ids,
            raw_audio=raw_wave_for_acoustic,
            neural_codec=self.neural_codec,
            num_coarse_quantizers=self.num_coarse_quantizers
        )
        assert exists(coarse_token_ids) and exists(fine_token_ids)

        return self.transformer_wrapper.forward(
            all_token_ids=[clap_token_ids, coarse_token_ids, fine_token_ids],
            return_loss=return_loss,
            **kwargs
        )


@beartype
class MusicLM(nn.Module):
    def __init__(
        self,
        *,
        wav2vec: Optional[Wav2Vec] = None,
        clap: ClapQuantized,
        neural_codec: NeuralCodec,
        semantic_transformer: TokenConditionedTransformer,
        coarse_transformer: TokenConditionedTransformer,
        fine_transformer: TokenConditionedTransformer,
        unique_consecutive=True
    ):
        super().__init__()

        assert semantic_transformer.token_sequences[1].codebook_size == coarse_transformer.token_sequences[1].codebook_size
        assert coarse_transformer.token_sequences[2].codebook_size == fine_transformer.token_sequences[2].codebook_size
        assert coarse_transformer.token_sequences[2].num_quantizers == fine_transformer.token_sequences[1].num_quantizers

        self.semantic = SemanticStage(
            semantic_transformer=semantic_transformer,
            wav2vec=wav2vec,
            clap=clap,
        )

        self.coarse = CoarseStage(
            coarse_transformer=coarse_transformer,
            wav2vec=wav2vec,
            clap=clap,
            neural_codec=neural_codec
        )        
        
        self.fine = FineStage(
            fine_transformer=fine_transformer,
            clap=clap,
            neural_codec=neural_codec
        )

        self.wav2vec = wav2vec
        self.clap = clap
        self.neural_codec = neural_codec

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    def forward(
        self,
        *,
        text: Optional[List[str]] = None,
        prime_wave=None,
        output_seconds=8,
        semantic_window_seconds=8,
        coarse_window_seconds=4,
        fine_window_seconds=2,
        semantic_steps_per_second=50, # Note: for Hubert its actually 50 * seconds - 1
        acoustic_steps_per_second=75, # 75 for encodec, 50 for soundstream
        return_coarse_generated_wave=False,
        mask_out_generated_fine_tokens=False,
        coarse_sliding_window_step_percent=0.5,
    ):
        assert exists(text), 'text needs to be passed in if one of the transformer requires conditioning'
        assert output_seconds <= semantic_window_seconds, 'no sliding semantic window (for now)'

        clap_token_ids = get_or_compute_clap_token_ids(None, self.clap, conditioning_audio=None, conditioning_text=text)

        all_semantic_token_ids = self.semantic.generate(
            clap_token_ids=clap_token_ids,
            max_time_steps=int(output_seconds * semantic_steps_per_second),
            include_eos_in_output=False,
            append_eos_to_conditioning_tokens=True,
        )

        # sliding windows of coarse window size 
        window_size = int(coarse_window_seconds * semantic_steps_per_second - 1)
        step_size = int(window_size * coarse_sliding_window_step_percent)
        all_semantic_token_ids = all_semantic_token_ids.unfold(1, window_size, step_size)
        all_semantic_token_ids = rearrange(all_semantic_token_ids, 'b n q w -> n b w q')

        all_coarse_token_ids = None
        for semantic_token_ids in all_semantic_token_ids:
            # TODO: pad to coarse_window_seconds if needed

            condition_length = int(coarse_window_seconds * acoustic_steps_per_second * (1 - coarse_sliding_window_step_percent))
            condition_coarse_token_ids = all_coarse_token_ids[:, -condition_length:] if exists(all_coarse_token_ids) else None

            pred_coarse_token_ids = self.coarse.generate(
                clap_token_ids=clap_token_ids,
                semantic_token_ids=semantic_token_ids,
                coarse_token_ids=condition_coarse_token_ids,
                max_time_steps=int(coarse_window_seconds * acoustic_steps_per_second),
                reconstruct_wave=False,
                include_eos_in_output=False,
                append_eos_to_conditioning_tokens=True,
                temperature=0.95,
            )

            if not exists(all_coarse_token_ids):
                all_coarse_token_ids = pred_coarse_token_ids
            else:
                pred_coarse_token_ids = pred_coarse_token_ids[:, condition_length:]
                all_coarse_token_ids = torch.cat([all_coarse_token_ids, pred_coarse_token_ids], dim=1)

        if return_coarse_generated_wave:
            wave = self.neural_codec.decode_from_codebook_indices(all_coarse_token_ids)
            wave = rearrange(wave, 'b 1 n -> b n')
            return wave.detach().cpu()

        # crop to fine window length and iterate 
        all_coarse_token_ids = torch.split(all_coarse_token_ids, fine_window_seconds * acoustic_steps_per_second, dim=1)

        generated_waves = []
        for coarse_token_ids in all_coarse_token_ids:
            generated_wave = self.fine.generate(
                clap_token_ids=clap_token_ids,
                coarse_token_ids=coarse_token_ids,
                max_time_steps=fine_window_seconds * acoustic_steps_per_second,
                reconstruct_wave=True,
                include_eos_in_output=False,
                append_eos_to_conditioning_tokens=True,
                temperature=0.4,
            )
            generated_waves.append(generated_wave)

        return torch.cat(generated_waves, dim=-1).detach().cpu()
