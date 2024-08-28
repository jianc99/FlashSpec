from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F

class QuestCache(DynamicCache):
    def __init__(self, page_size, topk)->None:
        super().__init__()
        self.page_size = page_size
        self.topk = topk
        self.budget = topk * page_size 
        self.num_pages = 0
        self.last_page_size = 0
        self.non_quest_layers = 2
        self.recent_key_cache: List[torch.Tensor] = []
        self.recent_value_cache: List[torch.Tensor] = []
        self.max_key: List[torch.Tensor] = []
        self.min_key: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def quest_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        assert len(self.key_cache) > layer_idx, f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}"
        if layer_idx < self.non_quest_layers:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        assert self._seen_tokens > self.budget, f"Seen tokens {self._seen_tokens} is less than budget {self.budget}"

        # assume that there are more than budget tokens in the prefill tokens
        # new tokens go into recent cache
        if layer_idx >= len(self.recent_key_cache) + self.non_quest_layers:
            self.recent_key_cache.append(key_states)
            self.recent_value_cache.append(value_states)
            recent_key_cache = key_states
            recent_value_cache = value_states
        else:
            recent_key_cache = self.recent_key_cache[layer_idx-self.non_quest_layers] 
            recent_value_cache = self.recent_value_cache[layer_idx-self.non_quest_layers] 
            recent_key_cache = torch.cat([recent_key_cache, key_states], dim=-2)
            recent_value_cache = torch.cat([recent_value_cache, value_states], dim=-2)
            self.recent_key_cache[layer_idx-self.non_quest_layers] = recent_key_cache
            self.recent_value_cache[layer_idx-self.non_quest_layers] = recent_value_cache

        B, num_heads, _, D = query_states.shape
        B, num_kv_heads, _, D = key_states.shape
        kv_repeats = num_heads // num_kv_heads

        min_key = repeat_kv(self.min_key[layer_idx-self.non_quest_layers], kv_repeats)   # [B, H, num_pages, D]
        max_key = repeat_kv(self.max_key[layer_idx-self.non_quest_layers], kv_repeats)   # [B, H, num_pages, D]

        min_product = min_key * query_states
        max_product = max_key * query_states

        heuristic = torch.maximum(min_product, max_product)
        heuristic = torch.sum(heuristic, dim=-1)    # [B, H, num_pages]
        heuristic = heuristic.view(B, num_kv_heads, kv_repeats, -1)
        heuristic = torch.sum(heuristic, dim=-2)    # [B, H_kv, num_pages] 

        topk_page_indices = torch.topk(heuristic, self.topk, dim=-1).indices    # [B, H_kv, topk]
        topk_page_indices = topk_page_indices[:, :, :, None, None].expand(-1, -1, -1, self.page_size, D)
        topk_keys = self.key_cache[layer_idx].gather(dim=2, index=topk_page_indices)    # [B, H_kv, topk, page_size, D]
        topk_values = self.value_cache[layer_idx].gather(dim=2, index=topk_page_indices)    # [B, H_kv, topk, page_size, D]

        topk_keys = topk_keys.view(B, num_kv_heads, -1, D)
        topk_values = topk_values.view(B, num_kv_heads, -1, D)

        k = torch.cat([topk_keys, recent_key_cache], dim=-2)
        v = torch.cat([topk_values, recent_value_cache], dim=-2)
        return k, v

    def build_pages(self):
        seq_len = self._seen_tokens
        self.num_pages = math.floor(seq_len / self.page_size)
        paged_seq_len = self.num_pages * self.page_size
        self.last_page_size = seq_len % self.page_size

        for i in range(self.non_quest_layers, len(self.key_cache)):
            key = self.key_cache[i]
            value = self.value_cache[i]
            B, num_heads, _, D = key.shape
            if self.last_page_size > 0:
                self.recent_key_cache.append(key[:, :, -self.last_page_size:].clone())
                self.recent_value_cache.append(value[:, :, -self.last_page_size:].clone())
            self.key_cache[i] = key[:, :, :paged_seq_len].reshape(B, num_heads, self.num_pages, self.page_size, D)
            self.value_cache[i] = value[:, :, :paged_seq_len].reshape(B, num_heads, self.num_pages, self.page_size, D)
            self.max_key.append(self.key_cache[i].max(dim=-2).values)
            self.min_key.append(self.key_cache[i].min(dim=-2).values)      

    @classmethod
    def from_fullcache(cls, page_size, topk, kv_cache):
        cache = cls(page_size, topk)
        if isinstance(kv_cache, Tuple):
            kv_cache = DynamicCache.from_legacy_cache(kv_cache)
        cache.key_cache = kv_cache.key_cache
        cache.value_cache = kv_cache.value_cache
        cache._seen_tokens = kv_cache.get_seq_length()
        cache.build_pages()
        return cache

