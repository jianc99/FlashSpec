import torch
from FlashSpec.Engine.model_draft import Transformer
from FlashSpec.Engine.utils import load_model_draft
import flashinfer

class LMBackend_Draft:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0") -> None:
        self.dtype = dtype
        self.device = device

        self.model_forward = {}
        self.model_forward[1] = lambda model, x, input_pos, cache_seqlens: model.forward_1(x, input_pos, cache_seqlens)
        self.model_forward[2] = lambda model, x, input_pos, cache_seqlens: model.forward_2(x, input_pos, cache_seqlens)
        self.prefill = lambda model, x, input_pos, cache_seqlens, is_last: model.prefill(x, input_pos, cache_seqlens, is_last)
        self.cachelens = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model_draft(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, kv_len: int = 512, llama3_1 = False):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.kv_len = kv_len

        # Init Draft Attention Backend(Flashinfer)
        self.decode1_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode2_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.arange(max_batch_size, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)

        self.decode1_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode1_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr.clone(), 
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr.clone(), 
                                                                              paged_kv_indices_buf=self.paged_kv_indices.clone(), 
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len.clone())
        
        self.decode2_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode2_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr.clone(), 
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr.clone(), 
                                                                              paged_kv_indices_buf=self.paged_kv_indices.clone(), 
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len.clone())
        
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer)
        self.decode_wrappers = [self.decode1_wrapper, self.decode2_wrapper]

        torch.library.define(
            "mylib::draft_decode1",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )
        @torch.library.impl("mylib::draft_decode1", "cuda")
        def target_decode(q, kv_cache):
            return self.decode1_wrapper.forward(
                q, kv_cache, pos_encoding_mode="NONE"
            )
        @torch.library.register_fake("mylib::draft_decode1")
        def target_decode_abstract(q, kv_cache):
            return torch.empty_like(q)
        
        torch.library.define(
            "mylib::draft_decode2",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )
        @torch.library.impl("mylib::draft_decode2", "cuda")
        def target_decode(q, kv_cache):
            return self.decode2_wrapper.forward(
                q, kv_cache, pos_encoding_mode="NONE"
            )
        @torch.library.register_fake("mylib::draft_decode2")
        def target_decode_abstract(q, kv_cache):
            return torch.empty_like(q)
        
        torch.library.define(
            "mylib::draft_prefill",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )
        @torch.library.impl("mylib::draft_prefill", "cuda")
        def target_prefill(q, kv_cache):
            return self.prefill_wrapper.forward(
                q, kv_cache, pos_encoding_mode="NONE"
            )
        @torch.library.register_fake("mylib::draft_prefill")
        def target_prefill_abstract(q, kv_cache):
            return torch.empty_like(q)

        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, kv_len=kv_len, is_llama3_1=llama3_1)

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)     
             
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False, cachelen_update = None):
            bsz, dec_len = input_ids.shape
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            if not benchmark:
                if cachelen_update == None:
                    self.cachelens += dec_len
                else:
                    self.cachelens += cachelen_update
            self.decode_wrappers[dec_len-1].begin_forward(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.cachelens,
                num_qo_heads=self.model.config.n_head, num_kv_heads=self.model.config.n_local_heads, head_dim=self.model.config.head_dim, page_size=self.max_length, q_data_type=self.dtype
            )
            logits = self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.flatten().clone(),
                input_pos=position_ids.flatten().clone(), 
                cache_seqlens= self.cachelens.clone())
            self.decode_wrappers[dec_len-1].end_forward()
            return logits.view(bsz, dec_len, -1)
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        chunk_size = 32
        num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
        for i in range(num_chunks):
            is_last = i == num_chunks-1
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_idx:end_idx]

            if end_idx > self.kv_len:
                chunk_position_ids = torch.arange(self.kv_len - chunk_input_ids.shape[1], self.kv_len, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
            else:
                chunk_position_ids = torch.arange(start_idx, end_idx, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
            

            chunk_len = end_idx - start_idx
            # chunk_cache_seqlens = start_idx

            self.prefill_wrapper.begin_forward(
            qo_indptr=self.qo_indptr*chunk_len,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            # paged_kv_last_page_len=chunk_cache_seqlens,
            num_qo_heads=self.model.config.n_head, num_kv_heads=self.model.config.n_local_heads, head_dim=self.model.config.head_dim, page_size=self.max_length, q_data_type=self.dtype
            )

            logits = self.prefill(
                model=self.model,
                x=chunk_input_ids,
                input_pos=chunk_position_ids,
                # cache_seqlens=chunk_cache_seqlens,
                is_last=is_last
            )

            self.prefill_wrapper.end_forward()
            
        self.cachelens += self.kv_len
        
        return logits.view(self.batch_size, -1, self.model.config.vocab_size)
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.kv_cache.zero_()
        self.cachelens.zero_()

    

