import torch
from FlashSpec.Engine.model import Transformer
from FlashSpec.Engine.utils import load_model
import flashinfer

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, cache_seqlens, position_ids: model(x, cache_seqlens, position_ids)
        self.prefill = lambda model, x, cache_seqlens, position_ids: model.prefill(x, cache_seqlens, position_ids)
        self.cachelens = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # Init Target Attention Backend(Flashinfer)
        self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.arange(max_batch_size, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
        self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr.clone(), 
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr.clone(), 
                                                                              paged_kv_indices_buf=self.paged_kv_indices.clone(), 
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len.clone())
        
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer)

        torch.library.define(
            "mylib::target_decode",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )
        @torch.library.impl("mylib::target_decode", "cuda")
        def target_decode(q, kv_cache):
            return self.decode_wrapper.forward(
                q, kv_cache, pos_encoding_mode="NONE"
            )
        @torch.library.register_fake("mylib::target_decode")
        def target_decode_abstract(q, kv_cache):
            return torch.empty_like(q)
        
        torch.library.define(
            "mylib::target_prefill",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )
        @torch.library.impl("mylib::target_prefill", "cuda")
        def target_prefill(q, kv_cache):
            return self.prefill_wrapper.forward(
                q, kv_cache, pos_encoding_mode="NONE"
            )
        @torch.library.register_fake("mylib::target_prefill")
        def target_prefill_abstract(q, kv_cache):
            return torch.empty_like(q)
        
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

    def compile(self):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)   
             
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            bsz, dec_len = input_ids.shape
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            if not benchmark:
                self.cachelens += dec_len
            self.decode_wrapper.begin_forward(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.cachelens,
                num_qo_heads=self.model.config.n_head, num_kv_heads=self.model.config.n_local_heads, head_dim=self.model.config.head_dim, page_size=self.max_length, q_data_type=self.dtype
            )
            logits = self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.flatten().clone(),
                cache_seqlens= self.cachelens.clone(),
                position_ids = position_ids.flatten().clone()) if dec_len in self.model_forward.keys() else self.model.forward(input_ids.clone(), self.cachelens.clone())
            self.decode_wrapper.end_forward()
            return logits.view(bsz, dec_len, -1)
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.cachelens.zero_()
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
        division = seq_len > 10000
        if division:
            chunk_size = 32
            num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, seq_len)
                
                chunk_input_ids = input_ids[:, start_idx:end_idx].flatten()
                chunk_len = end_idx - start_idx
                chunk_cache_seqlens = self.cachelens + start_idx + chunk_len

                chunk_position_ids = position_ids[:, start_idx:end_idx].flatten()

                self.prefill_wrapper.begin_forward(
                qo_indptr=self.qo_indptr*chunk_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=chunk_cache_seqlens,
                num_qo_heads=self.model.config.n_head, num_kv_heads=self.model.config.n_local_heads, head_dim=self.model.config.head_dim, page_size=self.max_length, q_data_type=self.dtype
                )
                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    cache_seqlens=chunk_cache_seqlens,
                    position_ids=chunk_position_ids,
                )
                self.prefill_wrapper.end_forward()

        else:
            self.prefill_wrapper.begin_forward(
                qo_indptr=self.qo_indptr*seq_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.cachelens + seq_len,
                num_qo_heads=self.model.config.n_head, num_kv_heads=self.model.config.n_local_heads, head_dim=self.model.config.head_dim, page_size=self.max_length, q_data_type=self.dtype
                )
            logits = self.prefill(
                model=self.model,
                x=input_ids.flatten(),
                cache_seqlens=self.cachelens + seq_len,
                position_ids= position_ids.flatten()
            )
            self.prefill_wrapper.end_forward()

        self.cachelens += seq_len
        
        return logits.view(self.batch_size, -1, self.model.config.vocab_size)
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.kv_cache.zero_()
            

    

