from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import tqdm
from audiolm_pytorch import (CoarseTransformer, FairseqVQWav2Vec,
                             FineTransformer, HubertWithKmeans,
                             SemanticTransformer, SoundStream)
from audiolm_pytorch.hubert_kmeans import HubertWithKmeans
from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
from beartype import beartype
from beartype.typing import List, Optional, Union
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from utils import (all_rows_have_eos_id, append_eos_id,
                   batch_unique_consecutive, default, eval_decorator, exists,
                   generate_mask_with_prob, gumbel_sample,
                   mask_out_after_eos_id, top_k)

# training wrappers


@beartype
class SemanticTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: SemanticTransformer,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        pad_id=-1,
        unique_consecutive=True,
        mask_prob=0.15
    ):
        super().__init__()
        self.wav2vec = wav2vec
        self.transformer = transformer
        assert not exists(
            self.wav2vec) or self.wav2vec.codebook_size == transformer.num_semantic_tokens, f'num_semantic_tokens on SemanticTransformer must be set to {self.wav2vec.codebook_size}'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id
        self.eos_id = transformer.eos_id
        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        max_length,
        text: Optional[List[str]] = None,
        text_embeds=None,
        prime_wave=None,
        prime_ids=None,
        batch_size=1,
        cond_scale=3,
        filter_thres=0.9,
        temperature=1.,
        # if doing hierarchical sampling, eos must be kept for an easy time
        include_eos_in_output=True,
        **kwargs
    ):
        device = self.device

        # derive wav2vec ids from the input wave

        if exists(prime_wave):
            assert not exists(prime_ids)
            assert exists(self.wav2vec)
            ids = self.wav2vec(prime_wave, flatten=False)
        elif exists(prime_ids):
            ids = prime_ids
        else:
            ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        if self.unique_consecutive:
            ids = batch_unique_consecutive(ids, pad_value=self.pad_id)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(
                    text, output_device=device)

        # start length and get running id output

        batch = ids.shape[0]
        start_length = ids.shape[-1]
        sample_semantic_ids = ids.clone()

        last_logit_indices = (ids != self.pad_id).sum(dim=-1).long()

        # sample from transformer

        for ind in tqdm(range(start_length, max_length), desc='generating semantic'):

            logits = self.transformer.forward_with_cond_scale(
                ids=sample_semantic_ids,
                text_embeds=text_embeds,
                **kwargs
            )

            last_logit_indices_expanded = repeat(
                last_logit_indices, 'b -> b 1 c', b=batch, c=logits.shape[-1])
            last_logits = logits.gather(1, last_logit_indices_expanded)

            last_logits = rearrange(last_logits, 'b 1 c -> b c')

            filtered_logits = top_k(last_logits, thres=filter_thres)
            sampled = gumbel_sample(
                filtered_logits, temperature=temperature, dim=-1)

            sampled = rearrange(sampled, 'b -> b 1')
            sample_semantic_ids = torch.cat(
                (sample_semantic_ids, sampled), dim=-1)

            if all_rows_have_eos_id(sample_semantic_ids, self.eos_id):
                break

            last_logit_indices += 1

        sample_semantic_ids = mask_out_after_eos_id(
            sample_semantic_ids, self.pad_id, keep_eos=False)

        return sample_semantic_ids

    def forward(
        self,
        *,
        semantic_token_ids=None,
        raw_wave=None,
        text=None,
        text_embeds=None,
        return_loss=False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(
            semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        if not exists(semantic_token_ids):
            assert exists(
                self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten=False)

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(
                semantic_token_ids, self.transformer.eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(
                semantic_token_ids, pad_value=self.pad_id)

        input_ids = semantic_token_ids
        if return_loss:
            input_ids = semantic_token_ids[:, :-1]

        self_attn_mask = None
        if self.mask_prob > 0. and self.training:
            self_attn_mask = generate_mask_with_prob(
                input_ids.shape, self.mask_prob, input_ids.device)

        logits = self.transformer(
            ids=input_ids,
            text=text,
            text_embeds=text_embeds,
            self_attn_mask=self_attn_mask,
            **kwargs
        )

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            semantic_token_ids,
            ignore_index=self.pad_id
        )

        return loss


@beartype
class CoarseTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        soundstream: Optional[SoundStream] = None,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]] = None,
        pad_id=-1,
        unique_consecutive=True,
        semantic_cross_entropy_loss_weight=1.,
        mask_prob=0.15
    ):
        super().__init__()
        self.soundstream = soundstream
        self.wav2vec = wav2vec

        self.transformer = transformer
        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight

        self.num_coarse_quantizers = transformer.num_coarse_quantizers
        self.semantic_eos_id = transformer.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        semantic_token_ids,
        text: Optional[List[str]] = None,
        text_embeds=None,
        max_time_steps=512,
        cond_scale=3.,
        filter_thres=0.9,
        temperature=1.,
        reconstruct_wave=False,
        **kwargs
    ):
        batch, device = semantic_token_ids.shape[0], self.device

        semantic_token_ids = semantic_token_ids.to(device)

        coarse_token_ids = torch.empty(
            (batch, 0), device=device, dtype=torch.long)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(
                    text, output_device=device)

        # initialize

        init_coarse_time_step = coarse_token_ids.shape[-1]
        sampled_coarse_token_ids = coarse_token_ids.clone()

        for time_step in tqdm(range(init_coarse_time_step, max_time_steps), desc='generating coarse'):
            for ind in range(self.num_coarse_quantizers):
                is_last_step = ind == (self.num_coarse_quantizers - 1)

                _, coarse_logits = self.transformer.forward_with_cond_scale(
                    coarse_token_ids=coarse_token_ids,
                    semantic_token_ids=semantic_token_ids,
                    text_embeds=text_embeds,
                    cond_scale=cond_scale,
                    return_only_coarse_logits=True,
                    **kwargs
                )

                last_coarse_logits = coarse_logits[:, -1]

                if not is_last_step:
                    # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval
                    last_coarse_logits[:, -1] = float('-inf')

                filtered_logits = top_k(last_coarse_logits, thres=filter_thres)
                sampled = gumbel_sample(
                    filtered_logits, temperature=temperature, dim=-1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat(
                    (sampled_coarse_token_ids, sampled), dim=-1)

        sampled_coarse_token_ids = mask_out_after_eos_id(
            sampled_coarse_token_ids, self.coarse_eos_id, keep_eos=False)
        sampled_coarse_token_ids = rearrange(
            sampled_coarse_token_ids, 'b (n q) -> b n q', q=self.num_coarse_quantizers)

        if not reconstruct_wave:
            return sampled_coarse_token_ids

        assert exists(self.soundstream)

        wav = self.soundstream.decode_from_codebook_indices(
            sampled_coarse_token_ids)
        return rearrange(wav, 'b 1 n -> b n')

    def forward(
        self,
        *,
        semantic_token_ids=None,
        raw_wave=None,
        raw_wave_for_soundstream=None,
        coarse_token_ids=None,
        return_loss=False,
        **kwargs
    ):
        assert exists(raw_wave) or exists(
            semantic_token_ids), 'either raw waveform (raw_wave) is given or semantic token ids are given (semantic_token_ids)'

        raw_wave_for_soundstream = default(raw_wave_for_soundstream, raw_wave)
        assert exists(raw_wave_for_soundstream) or exists(
            coarse_token_ids), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        assert not all(map(exists, (raw_wave, raw_wave_for_soundstream,
                       semantic_token_ids, coarse_token_ids)))

        if not exists(semantic_token_ids):
            assert exists(
                self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            semantic_token_ids = self.wav2vec(raw_wave, flatten=False)

        if not exists(coarse_token_ids):
            assert exists(
                self.soundstream), 'SoundStream must be provided if given raw wave for training'

            with torch.no_grad():
                self.soundstream.eval()
                _, indices, _ = self.soundstream(
                    raw_wave_for_soundstream, return_encoded=True)
                coarse_token_ids, _ = indices[...,
                                              :self.num_coarse_quantizers], indices[..., self.num_coarse_quantizers:]

        semantic_token_ids = rearrange(semantic_token_ids, 'b ... -> b (...)')
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        if self.training:
            semantic_token_ids = append_eos_id(
                semantic_token_ids, self.transformer.semantic_eos_id)
            coarse_token_ids = append_eos_id(
                coarse_token_ids, self.transformer.coarse_eos_id)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(
                semantic_token_ids, pad_value=self.pad_id)

        if return_loss:
            semantic_labels, coarse_labels = semantic_token_ids, coarse_token_ids.clone()
            coarse_token_ids = coarse_token_ids[:, :-1]

        # self attention mask would omit any padding and eos tokens in the semantic prime

        self_attn_mask = (semantic_token_ids != self.pad_id) & (
            semantic_token_ids != self.semantic_eos_id)
        semantic_token_ids = semantic_token_ids.masked_fill(~self_attn_mask, 0)

        coarse_token_len = coarse_token_ids.shape[-1]
        # attend to semantic bos and all coarse tokens
        self_attn_mask = F.pad(
            self_attn_mask, (1, coarse_token_len + 1), value=True)

        semantic_logits, coarse_logits = self.transformer(
            semantic_token_ids=semantic_token_ids,
            coarse_token_ids=coarse_token_ids,
            self_attn_mask=self_attn_mask,
            **kwargs
        )

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(
                self_attn_mask.shape, self.mask_prob, device=self_attn_mask.device)

        # whether to early return the logits

        if not return_loss:
            return semantic_logits, coarse_logits

        coarse_logits, semantic_logits = map(lambda t: rearrange(
            t, 'b n c -> b c n'), (coarse_logits, semantic_logits))

        if self.unique_consecutive:
            num_coarse_logits, num_semantic_logits = coarse_labels.numel(
            ), (semantic_labels != self.pad_id).sum()
        else:
            num_coarse_logits, num_semantic_logits = coarse_logits.shape[-1], semantic_logits.shape[-1]

        semantic_loss = 0.
        if self.semantic_cross_entropy_loss_weight > 0:
            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index=self.pad_id
            )

        coarse_loss = F.cross_entropy(
            coarse_logits,
            coarse_labels
        )

        return (
            semantic_loss * num_semantic_logits * self.semantic_cross_entropy_loss_weight +
            coarse_loss * num_coarse_logits
        ) / (num_semantic_logits + num_coarse_logits)


@beartype
class FineTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        transformer: FineTransformer,
        soundstream: Optional[SoundStream] = None,
        coarse_cross_entropy_loss_weight=1.,
        pad_id=-1,
        mask_prob=0.15
    ):
        super().__init__()
        self.soundstream = soundstream
        self.transformer = transformer

        self.num_fine_quantizers = transformer.num_fine_quantizers
        self.num_coarse_quantizers = transformer.num_coarse_quantizers
        self.eos_id = transformer.eos_id

        assert self.num_coarse_quantizers > 0

        self.pad_id = pad_id
        self.coarse_cross_entropy_loss_weight = coarse_cross_entropy_loss_weight

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        *,
        coarse_token_ids,
        text: Optional[List[str]] = None,
        text_embeds=None,
        cond_scale=3.,
        filter_thres=0.9,
        temperature=1.,
        reconstruct_wave=False,
        mask_out_generated_fine_tokens=False,
        **kwargs
    ):
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        batch, device = coarse_token_ids.shape[0], self.device

        coarse_token_ids = coarse_token_ids.to(device)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.no_grad():
                text_embeds = self.transformer.embed_text(
                    text, output_device=device)

        # initialize

        fine_token_ids = torch.empty(
            (batch, 0), device=device, dtype=torch.long)

        init_fine_time_step = fine_token_ids.shape[-1]
        max_time_steps = coarse_token_ids.shape[1] // self.num_coarse_quantizers

        sampled_fine_token_ids = fine_token_ids.clone()

        for time_step in tqdm(range(init_fine_time_step, max_time_steps), desc='generating fine'):
            for ind in range(self.num_fine_quantizers):
                is_last_step = ind == (self.num_fine_quantizers - 1)

                _, fine_logits = self.transformer.forward_with_cond_scale(
                    coarse_token_ids=coarse_token_ids,
                    fine_token_ids=fine_token_ids,
                    text_embeds=text_embeds,
                    cond_scale=cond_scale,
                    return_only_fine_logits=True,
                    **kwargs
                )

                last_fine_logits = fine_logits[:, -1]

                if not is_last_step:
                    # prevent from eos if not last quantizer step, but move this to masking logic within the transformer at some point, for both training and eval
                    last_fine_logits[:, -1] = float('-inf')

                filtered_logits = top_k(last_fine_logits, thres=filter_thres)
                sampled = gumbel_sample(
                    filtered_logits, temperature=temperature, dim=-1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_fine_token_ids = torch.cat(
                    (sampled_fine_token_ids, sampled), dim=-1)

        sampled_fine_token_ids = mask_out_after_eos_id(
            sampled_fine_token_ids, self.eos_id, keep_eos=False)

        # reshape coarse and fine tokens for quantization dimension

        sampled_fine_token_ids = rearrange(
            sampled_fine_token_ids, 'b (n q) -> b n q', q=self.num_fine_quantizers)
        coarse_token_ids = rearrange(
            coarse_token_ids, 'b (n q) -> b n q', q=self.num_coarse_quantizers)

        # whether to mask out fine token positions where the coarse token ids are all padding (variable lengthed training)

        if mask_out_generated_fine_tokens:
            pos_is_all_padding = (coarse_token_ids == self.pad_id).all(
                dim=-1, keepdim=True)
            seq_lengths = reduce(~pos_is_all_padding, 'b n 1 -> b', 'sum')

            sampled_fine_token_ids = sampled_fine_token_ids.masked_fill(
                pos_is_all_padding, self.pad_id)

        # if not reconstructing wave, return just the fine token ids

        if not reconstruct_wave:
            return sampled_fine_token_ids

        # reconstruct the wave using soundstream, concatting the fine and coarse token ids together first across quantization dimension

        assert exists(self.soundstream)

        coarse_and_fine_ids = torch.cat(
            (coarse_token_ids, sampled_fine_token_ids), dim=-1)

        wav = self.soundstream.decode_from_codebook_indices(
            coarse_and_fine_ids)
        return rearrange(wav, 'b 1 n -> b n')

    def forward(
        self,
        *,
        raw_wave=None,
        token_ids=None,
        coarse_token_ids=None,
        fine_token_ids=None,
        return_loss=False,
        **kwargs
    ):
        assert exists(raw_wave) ^ (exists(token_ids) ^ (exists(coarse_token_ids) and exists(fine_token_ids))
                                   ), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        if exists(raw_wave):
            assert exists(
                self.soundstream), 'SoundStream must be provided if given raw wave for training'

            with torch.no_grad():
                self.soundstream.eval()
                _, token_ids, _ = self.soundstream(
                    raw_wave, return_encoded=True)

        if exists(token_ids):
            coarse_token_ids, fine_token_ids = token_ids[...,
                                                         :self.num_coarse_quantizers], token_ids[..., self.num_coarse_quantizers:]

        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        fine_token_ids = rearrange(fine_token_ids, 'b ... -> b (...)')

        if self.training:
            coarse_token_ids = append_eos_id(
                coarse_token_ids, self.transformer.eos_id)
            fine_token_ids = append_eos_id(
                fine_token_ids, self.transformer.eos_id)

        if return_loss:
            coarse_labels, fine_labels = coarse_token_ids, fine_token_ids.clone()
            fine_token_ids = fine_token_ids[:, :-1]

        # do not attend to any of the coarse padding tokens or coarse end token either

        self_attn_mask = (coarse_token_ids != self.pad_id) & (
            coarse_token_ids != self.eos_id)
        coarse_token_ids = coarse_token_ids.masked_fill(~self_attn_mask, 0)

        fine_token_seq_len = fine_token_ids.shape[-1]
        self_attn_mask = F.pad(
            self_attn_mask, (1, fine_token_seq_len + 1), value=True)

        coarse_logits, fine_logits = self.transformer(
            coarse_token_ids=coarse_token_ids,
            fine_token_ids=fine_token_ids,
            self_attn_mask=self_attn_mask,
            **kwargs
        )

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(
                self_attn_mask.shape, self.mask_prob, device=self_attn_mask.device)

        # early return the logits

        if not return_loss:
            return coarse_logits, fine_logits

        coarse_logits, fine_logits = map(lambda t: rearrange(
            t, 'b n c -> b c n'), (coarse_logits, fine_logits))

        num_coarse_logits, num_fine_logits = coarse_logits.shape[-1], fine_logits.shape[-1]

        coarse_loss = 0.
        if self.coarse_cross_entropy_loss_weight > 0:
            coarse_loss = F.cross_entropy(
                coarse_logits,
                coarse_labels
            )

        fine_loss = F.cross_entropy(
            fine_logits,
            fine_labels
        )

        return (
            coarse_loss * num_coarse_logits * self.coarse_cross_entropy_loss_weight +
            fine_loss * num_fine_logits
        ) / (num_coarse_logits + num_fine_logits)

# audio LM


@beartype
class AudioLM(nn.Module):
    def __init__(
        self,
        *,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
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
        assert not (self.needs_text and not exists(
            text)), 'text needs to be passed in if one of the transformer requires conditioning'

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
