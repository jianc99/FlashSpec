import torch
import numpy as np
import random
from torch.nn.functional import softmax

def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    shape = logits.shape
    if top_p < 1.0:
        if len(shape)==3:
            batch_size, seq_len, voc_size = logits.size()
            logits = logits.reshape(-1, voc_size)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
        if len(shape)==3:
            logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def sample(logits, top_p, T):
    shape = logits.shape
    if len(shape)==3:
        batch_size, seq_len, _ = logits.size()
    else:
        batch_size, _ = logits.size()
        seq_len = 1
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cg_get_sampling_logits(logits :torch.Tensor, top_p:float, T: float):
    logits = logits.clone()
    batch_size, seq_len, voc_size = logits.size()
    logits = logits.reshape(-1, voc_size)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    filter[..., 0] = 0
    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
    logits[indices_to_remove] = float('-inf')
    logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def cg_sample(logits, top_p, T):
    batch_size, seq_len, _ = logits.size()
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cuda_graph_for_target_sample(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 3, batch_size=1, top_p = 0.9, T = 0.6):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
    def run(target_logits, top_p=None, T=None):
        static_sampling_logits.copy_(target_logits)
        graph.replay()
        return static_tokens.clone()
    return run

def sampling_argmax_batch(logits: torch.Tensor):
    return logits.topk(k=1, dim=-1).indices.flatten(start_dim=1).long()

def cuda_graph_for_sampling_argmax_batch(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 1, batch_size=1):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    return run

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    from FlashSpec.Engine.model import Transformer
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from FlashSpec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def load_model_draft(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    import FlashSpec.Engine.model_draft as draft
    with torch.device('meta'):
        model = draft.Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from FlashSpec.Engine.tp_draft import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()