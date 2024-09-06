from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F

class TopKCache(DynamicCache):
    def __init__(self, prefill: int, budget: int=1024):
        super().__init__()
        self.prefill = prefill
        self.budget = budget
        self.non_topk_layers = 2
        self.only_prefill_topk = True

    def topk_update(self,
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
        if layer_idx < self.non_topk_layers or query_states is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        assert query_states.shape[-2] == key_states.shape[-2], f"Query and key states must have the same sequence length"
        B, num_heads, L, D = query_states.shape
        B, num_kv_heads, _, D = key_states.shape
        kv_repeats = num_heads // num_kv_heads

        assert self._seen_tokens >= self.budget, f"Seen tokens {self._seen_tokens} is less than budget {self.budget}"
        assert L == 1, "only use topk during decoding"

        if self.only_prefill_topk:
            # Compute the attention scores
            prompt_key_states = self.key_cache[layer_idx][:, :, :self.prefill, :]
            prompt_key_states = repeat_kv(prompt_key_states, kv_repeats)
            S = prompt_key_states.shape[-2]
            
            attn_scores = torch.einsum("bhld,bhsd->bhls", query_states, prompt_key_states) 
            attn_scores = attn_scores.view(B, num_kv_heads, kv_repeats, L, S).sum(dim=2) # [B, H_kv, L, S]
            
            # Compute the top-k attention scores
            topk_indices = torch.topk(attn_scores, self.budget, dim=-1).indices # [B, H, L, budget]
            topk_indices = topk_indices.squeeze(-2) # [B, H, budget]
            topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            
            topk_prompt_keys = self.key_cache[layer_idx].gather(dim=2, index=topk_indices) # [B, H_kv, budget, D]
            topk_prompt_values = self.value_cache[layer_idx].gather(dim=2, index=topk_indices)

            # combine prompt_topk_keys with new keys
            topk_keys = torch.cat([topk_prompt_keys, self.key_cache[layer_idx][:, :, self.prefill:, :]], dim=-2)
            topk_values = torch.cat([topk_prompt_values, self.value_cache[layer_idx][:, :, self.prefill:, :]], dim=-2)

            return topk_keys, topk_values

        else:
            # Compute the attention scores
            expanded_key_states = repeat_kv(self.key_cache[layer_idx], kv_repeats)
            S = expanded_key_states.shape[-2]

            attn_scores = torch.einsum("bhld,bhsd->bhls", query_states, expanded_key_states)
            attn_scores = attn_scores.view(B, num_kv_heads, kv_repeats, L, S).sum(dim=2)

            # Compute the top-k attention scores
            topk_indices = torch.topk(attn_scores, self.budget, dim=-1).indices
            topk_indices = topk_indices.squeeze(-2)
            topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)

            topk_keys = self.key_cache[layer_idx].gather(dim=2, index=topk_indices)
            topk_values = self.value_cache[layer_idx].gather(dim=2, index=topk_indices)

            return topk_keys, topk_values

    
    @classmethod
    def from_fullcache(cls, kv_cache, prefill: int, budget: int=1024):
        cache = cls(prefill=prefill, budget=budget)
        if isinstance(kv_cache, Tuple):
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)
        cache.key_cache = kv_cache.key_cache
        cache.value_cache = kv_cache.value_cache
        cache._seen_tokens = kv_cache.get_seq_length()
        return cache