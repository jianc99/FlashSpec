import torch
from FlashSpec.Engine.model_draft import Transformer
from FlashSpec.Engine.utils import load_model_draft

class LMBackend_Draft:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_seqlens: model(x, input_pos, cache_seqlens)
        self.prefill = lambda model, x, input_pos, cache_seqlens: model.prefill(x, input_pos, cache_seqlens)
        self.cachelens = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model_draft(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, kv_len: int = 512):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.kv_len = kv_len
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, kv_len=kv_len)

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="reduce-overhead", fullgraph=True)      
             
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False, cachelen_update = None):
            dec_len = input_ids.shape[1]
            position_ids = torch.arange(self.kv_len - dec_len, self.kv_len, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
            # position_ids = torch.full((input_ids.shape[0],1), self.kv_len-1, device = self.device).long()
            logits = self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens= self.cachelens.clone()) if dec_len in self.model_forward.keys() else self.model.forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone())
            if not benchmark:
                if cachelen_update == None:
                    self.cachelens += dec_len
                else:
                    self.cachelens += cachelen_update
            return logits
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.cachelens.zero_()
        logits = None
        seq_len = input_ids.shape[1]
        chunk_size = 4
        num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            if end_idx > self.kv_len:
                chunk_position_ids = torch.arange(self.kv_len - chunk_input_ids.shape[1], self.kv_len, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
            else:
                chunk_position_ids = torch.arange(start_idx, end_idx, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
            chunk_cache_seqlens = start_idx

            logits = self.prefill(
                model=self.model,
                x=chunk_input_ids,
                input_pos=chunk_position_ids,
                cache_seqlens=chunk_cache_seqlens
            )
            
        self.cachelens += seq_len
        
        return logits
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.k_cache.zero_()
            b.attention.kv_cache.v_cache.zero_()

    

