from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F

class SnapKVCache(DynamicCache):
    def __init__(self, window_size: int=16, kernel_size: int=5, budget: int=1024):
        super().__init__()
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.budget = budget
        self.non_snap_layers = 2
        self.pooling_type = "max"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def snap_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        query_states: Optional[torch.Tensor]=None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        assert len(self.key_cache) > layer_idx, f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}"
        if layer_idx < self.non_snap_layers or query_states is None:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        assert query_states.shape[-2] == key_states.shape[-2], f"Query and key states must have the same sequence length"
        B, num_heads, L, D = query_states.shape
        B, num_kv_heads, _, D = key_states.shape
        kv_repeats = num_heads // num_kv_heads
        assert L > 1, "Can not snap in decoding stage"

        assert self._seen_tokens >= self.budget, f"Seen tokens {self._seen_tokens} is less than budget {self.budget}"

        # Compute the attention scores with BOTH past k and new k, and then only consider the past k
        past_key_states = self.key_cache[layer_idx]
        past_value_states = self.value_cache[layer_idx]
        ret_keys = torch.cat([past_key_states, key_states], dim=-2)
        ret_values = torch.cat([past_value_states, value_states], dim=-2)
        new_key_states = torch.cat([past_key_states, key_states], dim=-2)
        new_key_states = repeat_kv(new_key_states, kv_repeats)
        new_value_states = torch.cat([past_value_states, value_states], dim=-2)
        new_value_states = repeat_kv(new_value_states, kv_repeats)
        scale_factor = 1 / math.sqrt(D)
        
        S = new_key_states.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=key_states.dtype)
        # create a causal mask
        temp_mask = torch.zeros(L, S, dtype=torch.bool)
        temp_mask[:, -L:] = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        attn_bias = attn_bias.to(key_states.device)

        attn_scores = torch.einsum("bhld,bhsd->bhls", query_states, new_key_states) * scale_factor
        attn_scores += attn_bias
        attn_scores = F.softmax(attn_scores, dim=-1)

        # attn_output = torch.einsum("bhls,bhsd->bhld", attn_scores, new_value_states)
        
        vote = attn_scores[..., :-L].sum(dim=-2)    # summed across all the observed queries
        if self.pooling_type == "max":
            pool_vote = F.max_pool1d(vote, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        else:
            pool_vote = F.avg_pool1d(vote, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        pool_vote = pool_vote.view(B, num_kv_heads, kv_repeats, -1)
        pool_vote = pool_vote.sum(dim=-2)   # sum across kv_groups
        indices = pool_vote.topk(self.budget - L, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        
        selected_past_key = past_key_states.gather(dim=-2, index=indices)
        selected_past_value = past_value_states.gather(dim=-2, index=indices)
        self.key_cache[layer_idx] = torch.cat([selected_past_key, key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([selected_past_value, value_states], dim=-2)

        return ret_keys, ret_values
        # return attn_output


    @classmethod
    def from_fullcache(cls, kv_cache, window_size: int=16, kernel_size: int=5, budget: int=1024):
        cache = cls(window_size=window_size, kernel_size=kernel_size, budget=budget)
        if isinstance(kv_cache, Tuple):
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)
        cache.key_cache = kv_cache.key_cache
        cache.value_cache = kv_cache.value_cache
        cache._seen_tokens = kv_cache.get_seq_length()
        return cache