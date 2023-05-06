# Transformer implementation from https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/audiolm_pytorch.py

from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
import math

from .utils import default, exists, grad_shrink, l2norm

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# relative positional bias


class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers=3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    def forward(self, n, device=torch.device('cpu')):
        pos = torch.arange(n, device=device)
        rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
        rel_pos += (n - 1)

        x = torch.arange(-n + 1, n, device=device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        *,
        heads,
        num_buckets=32,
        max_distance=128,
        causal=True
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.causal = causal

        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(
        self,
        n,
        device=torch.device('cpu')
    ):
        pos = torch.arange(n, device=device)
        rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
        rel_pos = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)

        bias = self.relative_attention_bias(rel_pos)
        return rearrange(bias, 'i j h -> h i j')

# feedforward


class CausalDSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv1d(dim, dim, 3, bias=False, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = F.pad(x, (2, 0))
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def ConvFeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        CausalDSConv(inner_dim * 2),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

def FeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        non_causal_prefix=0,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        num_null_kv=0,
        dropout=0.1,
        scale=8,
        use_memory_efficient_attention=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        self.non_causal_prefix = non_causal_prefix
        self.dropout = dropout
        self.use_memory_efficient_attention = use_memory_efficient_attention
        if self.use_memory_efficient_attention and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")

        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout) if not self.use_memory_efficient_attention else None

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        prefix_context=None,
        prefix_context_mask=None
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value=0.)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b=b).unbind(dim=0)
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        # split for multi-headed attention

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # new technique, rmsnormed queries and keys, first used by 22B parameter model successfully https://arxiv.org/abs/2302.05442

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # attention

        if self.use_memory_efficient_attention:
            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)

                if exists(mask):
                    mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                    mask = rearrange(mask, 'b j -> b 1 1 j')
                    attn_bias = attn_bias.masked_fill(~mask, -torch.finfo(attn_bias.dtype).max)

                if self.causal:
                    i, j = attn_bias.shape[-2:]
                    causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)

                    if self.non_causal_prefix > 0:
                        causal_mask[:self.non_causal_prefix, :(self.non_causal_prefix + j - i)] = False

                    attn_bias = attn_bias.masked_fill(causal_mask, -torch.finfo(attn_bias.dtype).max)

            q = rearrange(q, 'b h n d -> b n h d')
            k = repeat(k, 'b n d -> b n h d', h=self.heads)
            v = repeat(v, 'b n d -> b n h d', h=self.heads)

            # compute attention
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)

            # merge heads
            out = rearrange(out, 'b n h d -> b n (h d)')

        else:
            sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
                sim = sim + attn_bias

            if exists(mask):
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)

                if self.non_causal_prefix > 0:
                    causal_mask[:self.non_causal_prefix, :(self.non_causal_prefix + j - i)] = False

                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)

            # aggregate
            out = einsum('b h i j, b j d -> b h i d', attn, v)

            # merge heads
            out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

# transformer


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context=None,
        cross_attend=False,
        attn_dropout=0.,
        ff_dropout=0.,
        use_conv_ff=True,
        grad_shrink_alpha=0.1,
        cond_as_self_attn_prefix=False,
        non_causal_prefix_size=0,
        relative_position_bias_type='continuous',
        **kwargs
    ):
        super().__init__()
        assert not (cross_attend and cond_as_self_attn_prefix)
        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha=grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        if relative_position_bias_type == 'continuous':
            self.rel_pos_bias = RelativePositionBias(dim=dim // 2, heads=heads)
        elif relative_position_bias_type == 't5':
            self.rel_pos_bias = T5RelativePositionBias(heads=heads, num_buckets=32, max_distance=128)
        elif relative_position_bias_type == 'none':
            self.rel_pos_bias = None
        else:
            raise ValueError(f'invalid relative position bias type: {relative_position_bias_type}')

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dropout=attn_dropout, causal=True, non_causal_prefix=non_causal_prefix_size, **kwargs),
                Attention(dim=dim, heads=heads, dropout=attn_dropout, dim_context=dim_context,
                          num_null_kv=1, norm_context=True, **kwargs) if cross_attend else None,
                ConvFeedForward(dim=dim, dropout=ff_dropout) if use_conv_ff else FeedForward(dim=dim, dropout=ff_dropout),
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        self_attn_mask=None,
        context=None,
        context_mask=None,
        attn_bias=None,
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(
            context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability
        x = self.grad_shrink(x)

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = self.rel_pos_bias(n, device = device) if exists(self.rel_pos_bias) else None

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context=context,
                prefix_context_mask=context_mask
            )

        for attn, cross_attn, ff in self.layers:
            x = attn(x, attn_bias=rel_pos_bias, mask=self_attn_mask, **self_attn_kwargs) + x

            if exists(cross_attn):
                assert exists(context)

                x = cross_attn(x, context=context, mask=context_mask) + x

            x = ff(x) + x

        return self.norm(x)
