from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F


class DynamicSnapKVCache(DynamicCache):
    def __init__(self, window_size: int=16, kernel_size: int=5, budget: int=1024):
        super().__init__()
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.budget = budget
        self.prompt_size = -1
        self.non_snap_layers = 2
        self.pooling_type = "avg"

        self.observed_query_cache: List[torch.Tensor] = []
        self.selected_key_cache: List[torch.Tensor] = []
        self.selected_value_cache: List[torch.Tensor] = []

        self.past_key_cache: List[torch.Tensor] = []
        self.past_value_cache: List[torch.Tensor] = []

    def snap_update(self,
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
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)    
        if layer_idx < self.non_snap_layers or query_states is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        elif len(self.selected_key_cache) <= layer_idx - self.non_snap_layers:
            ret_key_cache = self.key_cache[layer_idx]
            ret_value_cache = self.value_cache[layer_idx]
            # build snap cache for next gamma+1 generations
            self.build_snap_cache(layer_idx, query_states)
        else:
            ret_key_cache = torch.cat([self.selected_key_cache[layer_idx - self.non_snap_layers], self.key_cache[layer_idx]], dim=-2)
            ret_value_cache = torch.cat([self.selected_value_cache[layer_idx - self.non_snap_layers], self.value_cache[layer_idx]], dim=-2)
        
        # # build snap cache for next gamma+1 generations
        # self.build_snap_cache(layer_idx, query_states)

        return ret_key_cache, ret_value_cache


    def build_snap_cache(self,
        layer_idx: int,
        query_states: Optional[torch.Tensor]
    ) -> None:
        B, num_heads, L, D = query_states.shape

        if len(self.past_key_cache) <= layer_idx - self.non_snap_layers:
            key_cache = self.key_cache[layer_idx]
            self.past_key_cache.append(key_cache[...,:-L,:])   
            self.key_cache[layer_idx] = key_cache[...,-L:,:]
            self.past_value_cache.append(self.value_cache[layer_idx][...,:-L,:])
            self.value_cache[layer_idx] = self.value_cache[layer_idx][...,-L:,:]   
            observed_query = query_states
            self.observed_query_cache.append(observed_query)
        else:
            assert self.key_cache[layer_idx].shape[-2] == self.window_size + L, f"Key cache shape {self.key_cache[layer_idx].shape} does not match window size {self.window_size} and L {L}"
            past_key = self.past_key_cache[layer_idx - self.non_snap_layers]
            evicted_to_past_key = self.key_cache[layer_idx][...,:-self.window_size,:]
            self.key_cache[layer_idx] = self.key_cache[layer_idx][...,-self.window_size:,:]
            self.past_key_cache[layer_idx - self.non_snap_layers] = torch.cat([past_key, evicted_to_past_key], dim=-2)

            past_value = self.past_value_cache[layer_idx - self.non_snap_layers]
            evicted_to_past_value = self.value_cache[layer_idx][...,:-self.window_size,:]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][...,-self.window_size:,:]
            self.past_value_cache[layer_idx - self.non_snap_layers] = torch.cat([past_value, evicted_to_past_value], dim=-2)

            observed_query = self.observed_query_cache[layer_idx - self.non_snap_layers]
            observed_query = torch.cat([observed_query[...,-self.window_size+L:,:], query_states], dim=-2)
            self.observed_query_cache[layer_idx - self.non_snap_layers] = observed_query

        all_key_cache = torch.cat([self.past_key_cache[layer_idx - self.non_snap_layers], self.key_cache[layer_idx]], dim=-2)

        L = observed_query.shape[-2]
        num_kv_heads = all_key_cache.shape[1]
        kv_repeats = num_heads // num_kv_heads
        all_key_cache = repeat_kv(all_key_cache, kv_repeats)
        scale_factor = 1 / math.sqrt(D)

        S = all_key_cache.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=observed_query.dtype)
        # create a causal mask
        temp_mask = torch.zeros(L, S, dtype=torch.bool)
        temp_mask[:, -L:] = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        attn_bias = attn_bias.to(observed_query.device)

        attn_scores = torch.einsum("bhld,bhsd->bhls", observed_query, all_key_cache) * scale_factor
        attn_scores += attn_bias
        attn_scores = F.softmax(attn_scores, dim=-1)

        vote = attn_scores[..., :-self.window_size].sum(dim=-2)    # summed across all the observed queries
        if self.pooling_type == "max":
            pool_vote = F.max_pool1d(vote, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        else:
            pool_vote = F.avg_pool1d(vote, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        pool_vote = pool_vote.view(B, num_kv_heads, kv_repeats, -1)
        pool_vote = pool_vote.sum(dim=-2)   # sum across kv_groups
        indices = pool_vote.topk(self.budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        selected_past_key = all_key_cache.gather(dim=-2, index=indices)
        selected_past_value = self.past_value_cache[layer_idx - self.non_snap_layers].gather(dim=-2, index=indices)

        if len(self.selected_key_cache) <= layer_idx - self.non_snap_layers:
            self.selected_key_cache.append(selected_past_key)
            self.selected_value_cache.append(selected_past_value)
        else:
            self.selected_key_cache[layer_idx - self.non_snap_layers] = selected_past_key
            self.selected_value_cache[layer_idx - self.non_snap_layers] = selected_past_value
            

    @classmethod
    def from_fullcache(cls, kv_cache, window_size: int=16, kernel_size: int=5, budget: int=1024):
        cache = cls(window_size=window_size, kernel_size=kernel_size, budget=budget)
        if isinstance(kv_cache, Tuple):
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)
        cache.key_cache = kv_cache.key_cache
        cache.value_cache = kv_cache.value_cache
        cache._seen_tokens = kv_cache.get_seq_length()
        return cache


