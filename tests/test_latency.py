import sys
import time
from pathlib import Path
sys.path.append("..")
import torch
import torch._dynamo.config
import torch._inductor.config
import argparse
from FlashSpec.Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Your CLI description.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=256, help='max len')
parser.add_argument('--P', type=int, default=128, help='prefix len')
parser.add_argument('--T', type=int, default=1000, help='repeat times')
parser.add_argument('--declen_list', nargs='+', type=int, help='Group of dec len')
parser.add_argument('--rank_group', nargs='+', type=int, help='Group of ranks')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FlashSpec.Engine.tp import init_dist
use_tp = len(args.rank_group)>1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != 0:
        # only print on rank 0
        print = lambda *args, **kwargs: None
print(f"Using device={device}")

checkpoint_path = args.checkpoint_path
precision = torch.bfloat16
max_seq_length = args.M
max_batch_size = args.B
prefix_len = args.P
dec_list = args.declen_list

warm_up = 10
T = args.T

llm = LMBackend(dtype=precision, device=device, dec_list=dec_list)
llm.load_model(checkpoint_path, use_tp=use_tp, rank_group=args.rank_group, group = global_group)
if args.compile:
    llm.compile()
llm.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

prompt = torch.randint(low=3, high=30000, size=(max_batch_size, prefix_len), device=device)
llm.encode(input_ids=prompt)
print("finish encode")

for declen in dec_list:
    dec = torch.randint(low=3, high=30000, size=(max_batch_size, declen), device=device)
    with torch.inference_mode():
            for _ in range(warm_up):
                logits = llm.inference(input_ids=dec, benchmark=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(T):
                logits = llm.inference(input_ids=dec, benchmark=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
    print("Batch Size:{}, Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(max_batch_size, max_seq_length, declen, prefix_len, (t2 - t1)/ T))
