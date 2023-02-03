import itertools
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import tqdm
from audiolm_pytorch import (CoarseTransformer, CoarseTransformerWrapper,
                             FairseqVQWav2Vec, FineTransformer,
                             FineTransformerWrapper, HubertWithKmeans,
                             SemanticTransformer, SemanticTransformerWrapper,
                             SoundStream)
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans
from audiolm_pytorch.t5 import DEFAULT_T5_NAME
from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
from beartype import beartype
from beartype.typing import List, Optional, Union
from clap_quantized import ClapQuantized
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from transformer import Transformer
from utils import (all_rows_have_eos_id, get_embeds, append_eos_id,
                   batch_unique_consecutive, ceil_div, default, eval_decorator,
                   exists, generate_mask_with_prob, gumbel_sample,
                   mask_out_after_eos_id, round_down_nearest_multiple, top_k)


@dataclass
class TokenSequenceInfo():
    """
    Information about a type of token sequence that the TokenConditionedTransformer handles
    e.g. semantic tokens, coarse acoustic tokens, fine acoustic tokens, etc.
    """
    codebook_size: int
    num_quantizers: int    # e.g. 1 for semantic, Q for coarse acoustic, ...
    unique_consecutive: bool    # whether to remove unique consecutive tokens. see https://github.com/lucidrains/audiolm-pytorch/discussions/13#discussioncomment-4117105


class TokenConditionedTransformer(nn.Module):
    """
    Combination of the SemanticTransformer, CoarseTransformer and FineTransformer in lucidrain's AudioLM implementation,
    except that it is not tied to any specific type of token sequence. Instead, it can handle a variable number of
    token sequences, each with its own parameters. 
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
        ff_dropout=0.,
        has_condition=False,
        # t5_name = DEFAULT_T5_NAME,
        cond_as_self_attn_prefix=False,
        cond_drop_prob=0.5,
        grad_shrink_alpha=0.1,
        **kwargs
    ):
        super().__init__()

        self.token_sequences = token_sequences

        self.has_condition = has_condition
        self.cond_drop_prob = cond_drop_prob

        self.start_tokens, self.eos_ids, self.embeddings, self.logit_weights = [], [], [], []

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
                # text: Optional[List[str]] = None,
                # text_embeds = None,
                cond_drop_prob=None,
                return_only_final_seq_logits=False
                ):
        """
        all_token_ids: List of tensors containing token ids. Each element can either be 2 dimensional (batch_size, n_time_steps * num_quantizers) or 3 dimensional (batch_size, n_time_steps, num_quantizers)
                       Each element in list corresponds to one token sequence in self.token_sequences (e.g. semantic, coarse acoustic, fine acoustic, etc.)

        return_only_final_seq_logits: If True, only return logits for the final token sequence in self.token_sequences.
        """

        b, device = all_token_ids[0].shape[0], self.device

        all_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids)

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
            token_embeddings = get_embeds(embedding, token_ids, pad_id = -1) if sequence.unique_consecutive else embedding(token_ids)
            tokens.append(token_embeddings)
            start_tokens.append(repeat(start_token, 'd -> b 1 d', b=b))

            n_tokens = token_embeddings.shape[1]
            split_at.append(n_tokens if len(split_at) == 0 else split_at[-1] + n_tokens)

        tokens = list(itertools.chain(*zip(start_tokens, tokens)))  # [start_1, tokens_1, start_2, tokens_2, ...]
        tokens = torch.cat(tokens, dim=1)

        tokens = self.transformer(tokens, self_attn_mask=self_attn_mask)

        split_at = split_at[:-1]  # remove last element (total number of tokens)

        all_pred_tokens = torch.tensor_split(
            tokens, [sequence.num_quantizers for sequence in self.token_sequences], dim=1)

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
        cross_entropy_loss_weights: List[float] = None,
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
        max_time_steps=512,
        filter_thres=0.9,
        temperature=1.,
        **kwargs
    ):
        assert len(conditioning_token_ids) == len(self.token_sequences) - 1

        batch, device = conditioning_token_ids.shape[0], self.device

        conditioning_token_ids = conditioning_token_ids.to(device)

        pred_token_ids = torch.empty((batch, 0), device=device, dtype=torch.long)

        pred_sequence_info, pred_eos_id = self.token_sequences[-1], self.eos_ids[-1]

        # initialize

        init_pred_time_step = pred_token_ids.shape[-1]
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
            sampled_pred_token_ids, pred_eos_id, keep_eos=False)
        sampled_pred_token_ids = rearrange(
            sampled_pred_token_ids, 'b (n q) -> b n q', q=pred_sequence_info.num_quantizers)

        return sampled_pred_token_ids

    def forward(
        self,
        *,
        all_token_ids: List[torch.Tensor],
        return_loss=False,
        **kwargs
    ):
        assert len(all_token_ids) == len(self.token_sequences)

        batch, device = all_token_ids[0].shape[0], self.device

        all_token_ids = map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids)

        if self.training:
            all_token_ids = [append_eos_id(ids, eos_id) for ids, eos_id in zip(all_token_ids, self.eos_ids)]

        if self.unique_consecutive:
            for index, sequence_info in enumerate(self.token_sequences):
                if sequence_info.unique_consecutive:
                    all_token_ids[index] = batch_unique_consecutive(all_token_ids[index], pad_value=self.pad_id)

        if return_loss:
            all_labels = [ids for ids in all_token_ids[:-1]]
            all_labels[-1] = all_token_ids[-1].clone()

            all_token_ids[-1] = all_token_ids[-1][:, :-1]  # don't include eos in loss

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

        all_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), all_logits)

        total_logits = 0
        running_loss = 0.
        for logits, labels, num_all_logits, cross_entropy_loss_weight, sequence_info in zip(all_logits, all_labels, num_all_logits, self.cross_entropy_loss_weights, self.token_sequences):
            loss = 0.
            num_logits = 0

            if cross_entropy_loss_weight > 0 and exists(logits):
                num_logits = (labels != self.pad_id).sum() if self.unique_consecutive else labels.numel()

                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.pad_id if sequence_info.unique_consecutive else None
                )

            total_logits += num_logits
            running_loss += loss * num_logits * cross_entropy_loss_weight

        return running_loss / total_logits


@beartype
class MusicLM(nn.Module):
    def __init__(
        self,
        *,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
        clap: ClapQuantized,
        soundstream: SoundStream,
        semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        unique_consecutive=True
    ):
        super().__init__()

        assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers

        self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        self.needs_text = any(
            [self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])

        self.semantic = SemanticTransformerWrapper(
            wav2vec=wav2vec,
            transformer=semantic_transformer,
            unique_consecutive=unique_consecutive
        )

        self.coarse = CoarseTransformerWrapper(
            wav2vec=wav2vec,
            soundstream=soundstream,
            transformer=coarse_transformer,
            unique_consecutive=unique_consecutive
        )

        self.fine = FineTransformerWrapper(
            soundstream=soundstream,
            transformer=fine_transformer
        )

        self.clap = clap

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    def forward(
        self,
        *,
        batch_size=1,
        text: Optional[List[str]] = None,
        prime_wave=None,
        max_length=2048,
        return_coarse_generated_wave=False,
        mask_out_generated_fine_tokens=False
    ):
        """
        Given a condition text, generate a wave.
        Sample: a dict containing all the data of current sample.
        audio_data: a tensor of shape (T) containing audio data.
        max_len: the maximum length of audio data.
        data_truncating: the method of truncating data.
        data_filling: the method of filling data.
        audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
        """
        assert exists(text), 'text needs to be passed in if one of the transformer requires conditioning'

        if exists(prime_wave):
            prime_wave = prime_wave.to(self.device)

        semantic_token_ids = self.semantic.generate(
            text=text if self.semantic_has_condition else None,
            batch_size=batch_size,
            prime_wave=prime_wave,
            max_length=max_length
        )

        coarse_token_ids_or_recon_wave = self.coarse.generate(
            text=text if self.coarse_has_condition else None,
            semantic_token_ids=semantic_token_ids,
            reconstruct_wave=return_coarse_generated_wave
        )

        if return_coarse_generated_wave:
            return coarse_token_ids_or_recon_wave

        generated_wave = self.fine.generate(
            text=text if self.fine_has_condition else None,
            coarse_token_ids=coarse_token_ids_or_recon_wave,
            reconstruct_wave=True,
            mask_out_generated_fine_tokens=mask_out_generated_fine_tokens
        )

        return generated_wave
