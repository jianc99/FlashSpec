import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from FlashSpec.Engine.utils import setup_seed, get_sampling_logits
from FlashSpec.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from torch.nn.functional import softmax
from FlashSpec.Engine.backend import LMBackend

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--M', type=int, default=256, help='Maximum length.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='Dataset start index.')
parser.add_argument('--end', type=int, default=200, help='Dataset end index.')
parser.add_argument('--top_p', type=float, default=0.9, help='Target sample top_p.')
parser.add_argument('--temperature', type=float, default=0.6, help='Target sample temperature.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')

args = parser.parse_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from FlashSpec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        # only print on rank 0
        print = lambda *args, **kwargs: None
setup_seed(args.seed)
print(f"Using device={DEVICE}")
MAX_LEN = args.M
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.checkpoint_path
engine = LMBackend(dtype=DTYPE, device=DEVICE)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
if args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=False, drop_last=True)
num_eval_steps = len(dataloader)

total_time = 0.0
model_steps = 0
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch['input_ids'][..., :128].to(DEVICE)
    labels = batch['labels'][..., :128]
    terminate = False
    if (labels[:, -1] == -100)._is_any_true(): terminate = True
    output = input_ids.clone()
    prefix_len = input_ids.size(1)
    cache_lens = torch.zeros(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
    position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0).repeat(BATCH_SIZE,1)
    logits = engine.encode(input_ids=input_ids, position_ids=position_ids, cache_seqlens=cache_lens)
    seq_offset=prefix_len
    cache_lens += prefix_len
    logits = get_sampling_logits(logits=logits[:,-1], top_p=args.top_p, T=args.temperature, replicate=True)
    logits = softmax(logits / args.temperature, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(BATCH_SIZE, 1)
    output = torch.cat((output, next_tokens),dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    while output.size(1)<args.M and terminate == False:
        input_ids=next_tokens.clone()
        position_ids = torch.full((BATCH_SIZE,1),seq_offset, device=DEVICE)
        logits = engine.inference(input_ids=input_ids, position_ids=position_ids, cache_seqlens=cache_lens)
        logits = get_sampling_logits(logits=logits[:,-1], top_p=args.top_p, T=args.temperature, replicate=True)
        logits = softmax(logits / args.temperature, dim=-1)
        next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(BATCH_SIZE, 1)
        output = torch.cat((output, next_tokens),dim=-1)
        seq_offset += 1
        model_steps += 1
        cache_lens += 1
        if (next_tokens[:,-1] == 2)._is_any_true() or (next_tokens[:,-1] == 0)._is_any_true(): terminate = True
    torch.cuda.synchronize()
    t2=time.perf_counter()
    total_time += t2-t1
    print(f"Tokens per second :{total_time/model_steps}")
    if step == 0:
        total_time = 0.0
        model_steps = 0
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i]))
    if use_tp:
        dist.barrier()