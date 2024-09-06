import time
import torch
import numpy as np
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from transformers import PretrainedConfig

from FlashSpec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from FlashSpec.Data.data_converter import convert_pg19_dataset  #, LongBenchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from FlashSpec.Engine.snapkv.cache_dynamic_utils import DynamicSnapKVCache
from FlashSpec.Engine.snapkv.model_utils import enable_snap, disable_snap

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
import contextlib
import gc
from termcolor import colored
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Model name.')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset to use.')

parser.add_argument('--budget', type=int, default=256, help='KV Budget.')
parser.add_argument('--window_size', type=int, default=16, help='Window size.')
parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size.')
parser.add_argument('--gamma', type=int, default=5, help='Gamma.')

parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--gen_len', type=int, default=64, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')

args = parser.parse_args()

setup_seed(args.seed)
DEVICE = "cuda:0"
print(f"Using device={DEVICE}")
MAX_LEN_TARGET = args.prefix_len + args.gen_len + 1
DTYPE = torch.bfloat16
BATCH_SIZE = 1

P=0.9
T=0.01

############ Load model ############

model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                             device_map="auto",
                                             torch_dtype=torch.float16, 
                                             attn_implementation="flash_attention_2")

############ Load dataset ############
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")
eos_tokens = [eot_1, eot_2]

repeats = 20
no_runs = int(BATCH_SIZE*repeats)
if args.dataset == "pg19":
    dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len) #, end=no_runs)
else:
    dataset = LongBenchDataset(tokenizer=tokenizer, task="narrativeqa", seq_len=args.prefix_len) 
    eos_tokens = [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]] # although should not be useful for narrativeqa

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
try:
    num_eval_steps = min(20, len(dataloader))   # acc rate has low variance for pg19
except:
    num_eval_steps = 100

greedy = False
max_gen_toks = args.gen_len

######################## helper functions ############################
def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
    return logits

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    return residual


######################## eval loop ############################
agg_acc_rate = 0.
position_wise_acceptance_rates = torch.zeros(args.gen_len, device=DEVICE)
position_wise_samples = torch.zeros(args.gen_len, device=DEVICE)
total_samples = 0
iterative = args.prefix_len > 50_000

pbar = tqdm(enumerate(dataloader), total=num_eval_steps)
for step, batch in pbar:
    if step >= num_eval_steps:
        break
    if isinstance(batch, list):
        batch = batch[0]
    input_ids = batch.to(DEVICE)
    initial_len = input_ids.shape[1]
    tokens = input_ids.clone()    

    with torch.inference_mode():
        if iterative:
            # iterative prefill
            chunk_size = 1024
            past_key_values = DynamicCache()
            for i in range(0, initial_len, chunk_size):
                outputs = model(input_ids=input_ids[:, i:i+chunk_size], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1]
            logits = get_sampling_logits(logits, P, T, replicate=False)
            logits = torch.nn.functional.softmax(logits/T, dim=-1)
            # new_token = torch.multinomial(logits, num_samples=1)
            new_token = torch.argmax(logits, dim=-1, keepdim=True)    # greedy
            
            if new_token.item() in eos_tokens: break
            tokens = torch.cat([tokens, new_token], dim=1)

            for _ in range(max_gen_toks-1):
                outputs = model(input_ids=new_token, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1]
                logits = get_sampling_logits(logits, P, T, replicate=False)
                logits = torch.nn.functional.softmax(logits/T, dim=-1)
                # new_token = torch.multinomial(logits, num_samples=1)
                new_token = torch.argmax(logits, dim=-1, keepdim=True)    # greedy
                if new_token.item() in eos_tokens: break
                tokens = torch.cat([tokens, new_token], dim=1)
        
        else:
            if not greedy:
                outputs = model.generate(input_ids=input_ids, 
                                        do_sample=True,
                                        temperature=T,
                                        top_p=P,
                                        max_new_tokens=max_gen_toks, 
                                        return_dict_in_generate=True,
                                        eos_token_id=eos_tokens,)
            else:
                outputs = model.generate(input_ids=input_ids, 
                                        do_sample=False, 
                                        max_new_tokens=max_gen_toks, 
                                        eos_token_id=eos_tokens,
                                        return_dict_in_generate=True)

            tokens = outputs.sequences
        
        num_samples = tokens.shape[1] - initial_len 

        past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)   
        past_key_values.crop(initial_len)   # need to be cropped to get target and draft logits

        target_outputs = model(input_ids=tokens[:, initial_len:], past_key_values=past_key_values, use_cache=True)
        target_logits = target_outputs.logits
        past_key_values = target_outputs.past_key_values

        window_size = args.window_size
        past_key_values.crop(initial_len - window_size)
        past_key_values = DynamicSnapKVCache.from_fullcache(past_key_values, window_size, args.kernel_size, args.budget)

        enable_snap(model)

        # observe
        outputs = model(input_ids=tokens[:, initial_len-window_size:initial_len], past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        chunk_size = args.gamma + 1

        draft_logits = torch.zeros_like(target_logits)
        for i in range(0, num_samples, chunk_size):
            outputs = model(input_ids=tokens[:, i+initial_len:i+chunk_size+initial_len], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            draft_logits[:, i:i+chunk_size] = outputs.logits

        disable_snap(model)

        target_logits = get_sampling_logits(target_logits, P, T, replicate=False)

        target_proba = torch.nn.functional.softmax(target_logits/T, dim=-1).unsqueeze(-1)
        draft_proba = torch.nn.functional.softmax(draft_logits/T, dim=-1).unsqueeze(-1)

        probas = torch.cat([target_proba, draft_proba], dim=-1)
        probas = torch.min(probas, dim=-1).values
        acceptance_rate = probas.sum(dim=-1)    # [B, S]
        
        total_acceptance_rate = acceptance_rate.sum(dim=-1) 
        total_acceptance_rate = total_acceptance_rate.cumsum_(dim=0)

        if args.printoutput:
            # print last 10 input tokens in red color, and print new tokens in green color
            print(colored(tokenizer.decode(input_ids[0, -10:], skip_special_tokens=True), "red"), colored(tokenizer.decode(tokens[0, -num_samples:], skip_special_tokens=True), "green"))
            
    agg_acc_rate += total_acceptance_rate[0] 
    total_samples += num_samples
    pbar.set_description("acc rate: {:.2f}".format(agg_acc_rate / total_samples))

print("acc rate: ", agg_acc_rate / total_samples)