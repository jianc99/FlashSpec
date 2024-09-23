from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import math 
from FlashSpec.Engine.utils import custom_func, gqa_custom

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    scaling_factor:float = 1.0
    # llama 3.1 with high_freq_factor and low_freq_factor
    low_freq_factor: int = None # added new
    high_freq_factor: int = None  # added new
    original_max_position_embeddings: int = None   # added new

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        print(config)
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096, block_size = 4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "Wide-Sheared-LLaMA-543M": dict(block_size=4096, n_layer=3, n_head=32, n_local_heads=32, dim=4096, intermediate_size=11008, vocab_size=32000),
    "Wide-Sheared-LLaMA-290M": dict(block_size=4096, n_layer=1, n_head=32, n_local_heads=32, dim=4096, intermediate_size=11008, vocab_size=32000),
    "68m": dict(block_size=2048, n_layer=2, n_head=12, n_local_heads=12, dim=768, intermediate_size=3072, vocab_size=32000),
    "llama-160m": dict(block_size=2048, n_layer=12, n_head=12, n_local_heads=12, dim=768, intermediate_size=3072, vocab_size=32000),
    "1.3b": dict(block_size =2048, n_layer=24, n_head=16, n_local_heads=16, dim=2048, intermediate_size=5504, vocab_size=32000),
    "tinyllama": dict(block_size =2048, n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000),
    "Llama-3-8B-Instruct-Gradient-1048k": dict(block_size=1048576, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=3580165449),
    # new models
    # lmsys/vicuna-7b-v1.5-16k
    "vicuna-7b-v1.5-16k": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, scaling_factor=4.),
    'tiny-vicuna-1b': dict(block_size =2048, n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000), # same as tinyllama
    # togethercomputer/LLaMA-2-7B-32K
    'llama-2-7B-32K': dict(block_size=32768, n_layer=32, dim = 4096, vocab_size=32000,scaling_factor=8.),
    'tiny-vicuna-1b': dict(block_size=2048,n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000),
    # llama 3.1 with high_freq_factor and low_freq_factor
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0, scaling_factor=8,high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, 
                 snap_budget = 256, window_size = 64, kernel_size = 31, pooling_type="avg"):
        super().__init__()
        max_gen_tokens = 128
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('batch_indices',torch.arange(max_batch_size).unsqueeze(1))
        if snap_budget > 0:
            draft_cache_shape = (max_batch_size, snap_budget + window_size + max_gen_tokens, n_heads, head_dim)
            self.register_buffer('draft_k_cache', torch.zeros(draft_cache_shape, dtype=dtype))
            self.register_buffer('draft_v_cache', torch.zeros(draft_cache_shape, dtype=dtype))
        
        self.snap_budget = snap_budget
        if snap_budget > 0:
            # do NOT use snap kv for the first layer
            self.window_size = window_size
            self.kernel_size = kernel_size
            self.pooling_type = pooling_type
            
    def snap(self, q: Tensor, cache_len: int) -> None:
        if self.snap_budget <= 0:
            return
        
        window_size = self.window_size
        kernel_size = self.kernel_size
        snap_budget = self.snap_budget
        B, L, num_heads, D = q.shape
        num_kv_heads = self.k_cache.shape[2]
        assert L == window_size, f"Expected {window_size} queries for snap observation but got {L}"
        S = cache_len + window_size
        key_cache = self.k_cache[:, :S]
        kv_repeats = num_heads // num_kv_heads
        scale_factor = 1 / math.sqrt(D)

        attn_mask = torch.zeros(L, S, dtype=torch.bool, device=q.device)
        attn_mask[:, -L:] = torch.triu(torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1)
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        attn_bias.masked_fill_(attn_mask, float('-inf'))

        key_cache = key_cache.unsqueeze(-2).expand(-1, -1, -1, kv_repeats, -1)
        key_cache = key_cache.reshape(B, S, num_heads, D).transpose(1, 2).contiguous()
        attn_scores = torch.einsum('bhld,bhsd->bhls', q.transpose(1, 2), key_cache) * scale_factor
        attn_scores += attn_bias
        attn_scores = F.softmax(attn_scores, dim=-1)    # Do we really need softmax here?

        # sum across all the observed queries
        vote = attn_scores[..., :-window_size].sum(dim=-2) # (batch_size, nheads, seqlen)
        if self.pooling_type.lower() == "max":
            pool_vote = F.max_pool1d(vote, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        elif self.pooling_type.lower() == "avg":
            pool_vote = F.avg_pool1d(vote, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        else:
            pool_vote = vote

        pool_vote = pool_vote.view(B, num_kv_heads, kv_repeats, -1)
        pool_vote = pool_vote.sum(dim=-2) # (B, num_kv_heads, seqlen)

        topk_indices = pool_vote.topk(snap_budget, dim=-1).indices # (B, num_kv_heads, snap_budget)
        topk_indices = topk_indices.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, D)
        
        self.draft_k_cache[:, :snap_budget] = torch.gather(self.k_cache, 1, topk_indices)
        self.draft_k_cache[:, snap_budget: snap_budget + window_size] = self.k_cache[:, cache_len : cache_len + window_size]
        self.draft_v_cache[:, :snap_budget] = torch.gather(self.v_cache, 1, topk_indices)
        self.draft_v_cache[:, snap_budget: snap_budget + window_size] = self.v_cache[:, cache_len : cache_len + window_size]


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, 
                     snap_budget, window_size, kernel_size,
                     pooling_type="avg", num_non_snap_layers=0):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        # max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype

        for i, b in enumerate(self.layers):
            if i < num_non_snap_layers:
                layer_snap_budget = 0
            else:
                layer_snap_budget = snap_budget
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype,
                                           layer_snap_budget, window_size, kernel_size)
            
        if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
                                                  # new params
                                                  self.config.scaling_factor, self.config.low_freq_factor, self.config.high_freq_factor, self.config.original_max_position_embeddings)
        else:
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
                                                  # new params
                                                  self.config.scaling_factor)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, cache_seqlens)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def draft_forward(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor, draft_cachelens: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # import pdb; pdb.set_trace()
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.draft_forward(x, freqs_cis, cache_seqlens, draft_cachelens)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def prefill(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor, snap: bool=False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(x, freqs_cis, cache_seqlens, snap=snap)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache_seqlens)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, snap: bool) -> Tensor:
        h = x + self.attention.prefill(self.attention_norm(x), freqs_cis, cache_seqlens, snap)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def draft_forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, draft_cachelens: Tensor) -> Tensor:
        h = x + self.attention.draft_forward(self.attention_norm(x), freqs_cis, cache_seqlens, draft_cachelens)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.process_group = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        # if self.n_head == self.n_local_heads:
        #     self._attn = torch.ops.mylib.custom_func
        # else:
        #     self._attn = torch.ops.mylib.gqa_custom
        self._attn = torch.ops.mylib.custom_func

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        k_cache, v_cache = self.kv_cache.k_cache, self.kv_cache.v_cache

        y = self._attn(q, k_cache, v_cache, k, v, cache_seqlens)

        y = y.contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y

    def draft_forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, draft_cachelens: Tensor) -> Tensor:
        # cache_seqlens is the length of the draft cache only
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # TODO: more memory efficient imnplementation
        # narrow ensures in-place slicing so that the cache is updated by flash_attn
        # k_cache = self.kv_cache.k_cache.narrow(1, self.draft_kv_start_idx, cache_seqlens)   
        # v_cache = self.kv_cache.v_cache.narrow(1, self.draft_kv_start_idx, cache_seqlens)

        if self.kv_cache.snap_budget > 0:
            k_cache, v_cache = self.kv_cache.draft_k_cache, self.kv_cache.draft_v_cache
            y = self._attn(q, k_cache, v_cache, k, v, draft_cachelens)
        else:
            k_cache, v_cache = self.kv_cache.k_cache, self.kv_cache.v_cache
            y = self._attn(q, k_cache, v_cache, k, v, cache_seqlens)

        y = y.contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y


    def prefill(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, snap: bool=False) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache.k_cache, self.kv_cache.v_cache

        y = self._attn(q, k_cache, v_cache, k, v, cache_seqlens)

        if snap:
            # KV cache has already been updated by flash_attn
            self.kv_cache.snap(q, cache_seqlens[0].item())

        y = y.contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _compute_llama3_parameters(inv_freq, old_context_len=8192, scaling_factor=8,low_freq_factor=1,high_freq_factor=4):
    """
    To be used for llama 3.1 models
        - borrowing the logic from: https://github.com/huggingface/transformers/blob/c85510f958e6955d88ea1bafb4f320074bfbd0c1/src/transformers/modeling_rope_utils.py
        - source: _compute_llama3_parameters in modeling_rope_utils.py
    """
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scaling_factor + smooth * freq)
    inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
    return inv_freq

# def precompute_freqs_cis(
#     seq_len: int, n_elem: int, base: int = 10000,
#     dtype: torch.dtype = torch.bfloat16,
#     scaling_factor = 1
# ) -> Tensor:
#     freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
#     freqs /= scaling_factor
#     t = torch.arange(seq_len, device=freqs.device, dtype=freqs.dtype)
#     # t /=scaling_factor
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
#     return cache.to(dtype=dtype)

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    scaling_factor: float = 1.0, # added new 
    low_freq_factor: int = None, # added new
    high_freq_factor: int = None, # added new
    original_max_position_embeddings: int = None, # added new
) -> Tensor:
    print(f"draft: seq_len: {seq_len}, n_elem: {n_elem}, base: {base}, dtype: {dtype}, scaling_factor: {scaling_factor}, low_freq_factor: {low_freq_factor}, high_freq_factor: {high_freq_factor}, original_max_position_embeddings: {original_max_position_embeddings}")

    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    
    if (low_freq_factor is not None) and (high_freq_factor is not None):
        freqs = _compute_llama3_parameters(freqs, original_max_position_embeddings, scaling_factor, low_freq_factor,high_freq_factor)
    else:
        freqs /= scaling_factor
    t = torch.arange(seq_len, device=freqs.device, dtype=freqs.dtype)
    # t /=scaling_factor
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(x.shape[0], xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)