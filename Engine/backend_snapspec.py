import torch
from FlashSpec.Engine.model_snapspec import Transformer
from FlashSpec.Engine.utils import load_model_snapspec  # TODO: check

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1], draft_dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        self.draft_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_seqlens: model(x, input_pos, cache_seqlens)
        for dec_len in draft_dec_list:
            if dec_len == 0: continue
            self.draft_forward[dec_len] = lambda model, x, input_pos, cache_seqlens, draft_cachelens: model.draft_forward(x, input_pos, cache_seqlens, draft_cachelens)
        self.prefill = lambda model, x, input_pos, cache_seqlens, snap: model.prefill(x, input_pos, cache_seqlens, snap)
        self.cachelens = None
        self.draft_cachelens = None
        self.snap_budget = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model_snapspec(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, buffer: int = 0, 
                     snap_budget: int = 256, window_size: int = 32, kernel_size: int = 5, pooling_type: str = "avg"):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, 
                                    snap_budget=snap_budget, window_size=window_size, kernel_size=kernel_size, pooling_type=pooling_type)
        self.snap_budget = snap_budget
        self.window_size = window_size

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)
        for key in self.draft_forward.keys():
            self.draft_forward[key] = torch.compile(self.draft_forward[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="reduce-overhead", fullgraph=True)      

    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            # import pdb; pdb.set_trace()
            logits = self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens= self.cachelens.clone()) if dec_len in self.model_forward.keys() else self.model.forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone())
            if not benchmark:
                self.cachelens += dec_len

            return logits


    @torch.inference_mode()
    def draft_inference(self, input_ids: torch.LongTensor, benchmark = False, cachelen_update = None):
            dec_len = input_ids.shape[1]
            position_ids = self.cachelens.view(-1,1) - dec_len + 1 + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            # import pdb; pdb.set_trace()
            logits = self.draft_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens= self.cachelens.clone(),
                draft_cachelens= self.draft_cachelens.clone()) if dec_len in self.draft_forward.keys() else self.model.draft_forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone(), self.draft_cachelens.clone())
            if not benchmark:
                if cachelen_update == None:
                    self.draft_cachelens += dec_len
                    self.cachelens += dec_len   # for non-snap layers
                else:
                    self.draft_cachelens += cachelen_update
                    self.cachelens += cachelen_update   # for non-snap layers
            return logits
    

    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.cachelens.zero_()
        self.draft_cachelens.zero_()
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
        division = seq_len > 1000
        context_len = seq_len - self.window_size

        # first prefill the context -> then snap observe using last window_size tokens
        if division:
            chunk_size = 32
            num_chunks = (context_len + chunk_size - 1) // chunk_size  # Ceil division
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, context_len)
                
                chunk_input_ids = input_ids[:, start_idx:end_idx]
                chunk_position_ids = position_ids[:, start_idx:end_idx]
                chunk_cache_seqlens = self.cachelens + start_idx

                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=chunk_position_ids,
                    cache_seqlens=chunk_cache_seqlens,
                    snap=False
                )
        else:
            logits = self.prefill(
                model=self.model,
                x=input_ids[:, :context_len],
                input_pos=position_ids[:, :context_len],
                cache_seqlens=self.cachelens,
                snap=False
            )

        # snap observe and create snap kv cache
        logits = self.prefill(
            model=self.model,
            x=input_ids[:, context_len:],
            input_pos=position_ids[:, context_len:],
            cache_seqlens=self.cachelens + context_len,
            snap=True
        )

        self.cachelens += seq_len
        # build snap cache
        self.draft_cachelens += self.snap_budget + self.window_size

        return logits


    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.k_cache.zero_()
            b.attention.kv_cache.v_cache.zero_()